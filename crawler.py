"""Crawl festival info that is later searchable and filterable on a frontend."""

import requests
import json
import re
import os
import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import concurrent.futures
from bs4 import BeautifulSoup
from tqdm import tqdm
from tqdm.asyncio import tqdm as tqdm_asyncio
from typing import List, Tuple, Optional
from urllib.parse import urljoin, urlparse
import asyncio
from playwright.async_api import (
    async_playwright,
    Error as PlaywrightError,
    TimeoutError as PlaywrightTimeoutError,
)
import google.generativeai as genai
from google.generativeai import GenerationConfig
from google.api_core.exceptions import ResourceExhausted, DeadlineExceeded
from dotenv import load_dotenv
from utils import get_lat_long

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
MAX_CONCURRENT_PAGES = 10
MAX_CONCURRENT_TASKS = 10
BLOCK_RESOURCE_TYPES = ["image", "stylesheet", "font", "media"]
MAX_RETRIES = 2 * 30
NAV_TIMEOUT_MS = 30000
EXTRA_WAIT_MS = 2000


def get_request(url: str) -> requests.Response:
    """Execute GET request using requests package and throw an error if not HTTP code 200.

    Args:
        url (str): url to GET

    Returns:
        requests.Response: response

    """
    response = requests.get(url)
    response.raise_for_status()

    return response


async def fetch_page_content(
    browser, url: str, semaphore: asyncio.Semaphore
) -> Tuple[str, Optional[BeautifulSoup]]:
    """Fetch a single page using Playwright.

    Args:
        browser: An active Playwright browser instance.
        url: The URL to fetch.
        semaphore: An asyncio.Semaphore to limit concurrency.

    Returns:
        A tuple: (url, BeautifulSoup object or None if failed).

    """
    page = None
    context = None
    soup = None

    # facebook pages do not lead to any info since there is always the login screen.
    # if the page does not have its own website they do not deserve all bands to be listed.
    if "facebook" in url:
        return url, soup

    async with semaphore:
        try:
            # Create a new isolated browser context for each request
            context = await browser.new_context(
                user_agent=USER_AGENT,
            )
            page = await context.new_page()

            # --- Block unnecessary resources ---
            async def handle_route(route):
                if route.request.resource_type in BLOCK_RESOURCE_TYPES:
                    try:
                        await route.abort()
                    except PlaywrightError as e:
                        print(
                            f"Warning: Could not abort request {route.request.url}: {e}"
                        )
                else:
                    await route.continue_()

            await page.route("**/*", handle_route)

            # --- Navigate and wait for domcontentloaded (or timeout) ---
            try:
                await page.goto(
                    url, wait_until="domcontentloaded", timeout=NAV_TIMEOUT_MS
                )
            except PlaywrightTimeoutError:
                print(f"Timeout waiting for domcontentloaded: {url}")
            except Exception as e:
                print(f"Navigation error for {url}: {type(e).__name__} - {e}")

            if page:
                await page.wait_for_timeout(EXTRA_WAIT_MS)
            else:
                print(f"Skipping extra wait as page was not created for {url}")

            # --- Get page content ---
            if page:
                html_content = await page.content()
            else:
                html_content = None
                print(f"Skipping content retrieval as page was not created for {url}")

            if html_content:
                soup = BeautifulSoup(html_content, "html.parser")
            else:
                print("Got no html content!")

        except PlaywrightError as e:
            print(f"Playwright setup error for {url}: {e}")
        except Exception as e:
            print(
                f"Unexpected error during fetch process for {url}: {e}", exc_info=True
            )
        finally:
            # --- Cleanup ---
            if page:
                try:
                    await page.close()
                except Exception as e:
                    print(f"Error closing page for {url}: {e}")
            if context:
                try:
                    await context.close()
                except Exception as e:
                    print(f"Error closing context for {url}: {e}")

            return url, soup


async def run_parallel_fetch(
    urls: List[str],
) -> List[Tuple[str, Optional[BeautifulSoup]]]:
    """Fetch multiple URLs in parallel using Playwright, showing progress with tqdm.

    Args:
        urls: A list of URLs to fetch.

    Returns:
        A list of tuples: [(url, BeautifulSoup object or None), ...],
        in the order of completion.

    """
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)
    print(
        f"Setting up Playwright and browser (max concurrent tasks: {MAX_CONCURRENT_TASKS})..."
    )

    async with async_playwright() as p:
        browser = await p.firefox.launch(headless=True)
        print(f"Browser launched: {browser.browser_type.name} {browser.version}")

        tasks = [fetch_page_content(browser, url, semaphore) for url in urls]

        print(f"Starting parallel fetch for {len(urls)} URLs...")

        results_in_completion_order = []
        for future in tqdm_asyncio(
            asyncio.as_completed(tasks),
            total=len(urls),
            desc="Fetching URLs",
            unit="site",
        ):
            try:
                result = await future
                results_in_completion_order.append(result)
            except Exception as e:
                print(
                    f"\nError retrieving result from a task future: {e}", exc_info=True
                )

        print("\nAll fetch tasks completed.")
        await browser.close()
        print("Browser closed.")

    return results_in_completion_order


def gemini_generate(prompt: str) -> str:
    """Run Gemini API using the passed prompt.

    Args:
        prompt (str): prompt to pass to Gemini API

    Returns:
        str: response. Since structured output is used, this contains json as a string.

    """
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel("gemini-2.0-flash-lite")

    def safe_generate():
        return model.generate_content(
            prompt,
            generation_config=GenerationConfig(response_mime_type="application/json"),
            request_options={"timeout": 60 * 30},  # 30min
        )

    for attempt in range(MAX_RETRIES):
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(safe_generate)
                response = future.result(timeout=60 * 31)  # 31min
                return response.text
        except concurrent.futures.TimeoutError:
            print(f"Timeout on attempt {attempt + 1}")
        except ResourceExhausted as e:
            pass  # this happens way too often. floods output
            # print("ResourceExhausted.")
        except DeadlineExceeded as e:
            print("Got DeadlineExceeded:", e)
        time.sleep(25)

    return None


def format_duration(seconds: float) -> str:
    """Format duration from seconds to hh:mm:ss.

    Args:
        seconds (float): duration in seconds

    Returns:
        str: duration formatted

    """
    return str(timedelta(seconds=int(seconds)))


def get_festival_list() -> List[dict]:
    """Crawl name and link to a dedicated page for all festivals listed on festivalticker.de.

    Returns:
        List[dict]: list of festivals

    """
    url = "https://www.festivalticker.de/alle-festivals/"
    response = get_request(url)

    soup = BeautifulSoup(response.content, "html.parser")
    events = soup.body.find_all("tbody", class_="vevent")
    parsed_events = []
    for event in events:
        tr = event.find_next("tr")
        # do not consider line-through elements
        if "style" in tr.attrs.keys() and "text-decoration:line-through;" in tr.get(
            "style"
        ):
            continue
        link = event.find("a")
        name = link.get_text()
        festival_ticker_url = link.get("href")
        parsed_events.append({"name": name, "festival_ticker_url": festival_ticker_url})

    return parsed_events


def crawl_festival_from_festival_tickers_dedicated_page(url: str):
    """Crawl more detailed information about a single festival from festivalticker's dedicated page for that festival.

    Args:
        url (str): festivalticker url of that festival

    """

    def extract_date(info):
        date_p = info.find_next("p").get_text().strip().split("\n")[0].strip()
        date_p_split = date_p.split(" ")
        start_date = date_p_split[1]
        end_date = date_p_split[3] if len(date_p_split) > 2 else None

        return start_date, end_date

    def extract_genres(info):
        genre_raw = (
            info.find_next("table")
            .find_next("tr")
            .find_all_next("td")[1]
            .get_text()
            .strip()
        )

        # if 'show more genres' was available: genres='<short-list>... mehr\n<long-list>', only keep <long-list>
        show_more_text = "... mehr\n"
        if show_more_text in genre_raw:
            idx = genre_raw.find(show_more_text)
            genre_raw = genre_raw[idx + len(show_more_text) :]

        genres = genre_raw.split(", ")
        # get rid of trailing garbage
        if genres[-1][-4:] == " ...":
            genres[-1] = genres[-1][:-4]
        elif genres[-1][-10:] == " ... close":
            genres[-1] = genres[-1][:-10]

        return genres

    def extract_festival_url(info):
        anchor_tag = info.find_all_next("table")[2].find("a")
        return anchor_tag.get("href")

    def extract_bands(info):
        website_bands_rows = info.find_all_next("table")[2].find_all_next("tr")
        if len(website_bands_rows) <= 1:
            return []

        tr_index_of_bands = None
        for i, bands in enumerate(website_bands_rows):
            bands = bands.text.strip()
            if bands.startswith("Bands:"):
                tr_index_of_bands = i
                break
        if tr_index_of_bands is None:
            return []

        bands = bands.split("\n")
        if len(bands) > 1:
            bands = bands[1]
        else:  # in case the "Bands:" heading is not the same row as the bands themselves
            tr_index_of_bands += 1
            bands = website_bands_rows[tr_index_of_bands].text.strip()

        bands = bands.split(", ")
        bands[0] = bands[0].strip()  # remove leading whitespaces
        # if 'und weitere': remove it
        if bands[-1].endswith("\rund weitere"):
            bands[-1] = bands[-1].split("\r")[0]

        return bands

    def extract_location(info):
        location = {}
        features = {
            "Location:": "location",
            "Plz:": "zip_code",
            "Ort:": "city",
            "Strasse:": "address_line_1",
            "Land:": "country",
        }

        location_div = info.find_next("div", {"class": "location"})
        feature_keys = features.keys()
        for tr in location_div.find_all("tr"):
            split = tr.text.strip().split("\n")
            if (feature := split[0].strip()) in feature_keys and len(split) > 1:
                location[features[feature]] = split[1]

        query = ", ".join(
            [location[feature] for feature in features.values() if feature in location]
        )
        latitude, longitude = get_lat_long(query)
        location["latitude"] = latitude
        location["longitude"] = longitude

        return location

    response = get_request(url)
    soup = BeautifulSoup(response.content, "html.parser")

    main_info = soup.body.find_next("table")
    main_info = main_info.find_next("td").find_next("table").find_next("td")

    start_date, end_date = extract_date(main_info)
    genres = extract_genres(main_info)
    url = extract_festival_url(main_info)
    bands = extract_bands(main_info)
    location = extract_location(main_info)

    return {
        "start_date": start_date,
        "end_date": end_date,
        "genres": genres,
        "url": url,
        "bands": bands,
        "location": location,
    }


def crawl_festivals_from_festival_ticker() -> List[dict]:
    """Crawl all festivals from festivalticker.de with its important information.

    Returns:
        List[dict]: list of festivals

    """
    events = get_festival_list()

    events_updated = []
    for event in tqdm(
        events, desc="Scraping info of festival ticker's dedicated pages"
    ):
        info = crawl_festival_from_festival_tickers_dedicated_page(
            event["festival_ticker_url"]
        )
        event.update(info)
        events_updated.append(event)

    return events_updated


def find_line_up_link_for_a_single_festival(festival: dict) -> str:
    """Get the link to the festival's line up page. Defaults to homepage.

    Args:
        festival (dict): festival to find the link for

    Returns:
        str: url

    """

    def get_absolute_link(base: str, a: str):
        parsed_url = urlparse(a)
        if parsed_url.scheme == "https" or parsed_url.scheme == "http":
            return a
        else:
            return urljoin(base, a)

    slug_attempts = ["line-up", "bands", "lineup", "acts", "artists"]
    base_url = festival["url"]

    # try urls as good-luck-guesses
    for slug in slug_attempts:
        url = urljoin(base_url, slug)
        try:
            response = requests.get(url, timeout=5.0)
            if response.status_code == 200:
                return url
        except:
            pass

    try:
        response = requests.get(base_url, timeout=10.0)
        soup = BeautifulSoup(response.content, "html.parser")
    except:
        return base_url

    # find slug in href of anchor tags that match the slug
    for slug in slug_attempts:
        pattern = re.compile(f"([^a-zA-Z])*{slug}(.|\n)*", flags=re.IGNORECASE)
        anchor_tags = soup.find_all("a", href=True)
        for a in anchor_tags:
            if pattern.match(a["href"]):
                return get_absolute_link(base_url, a["href"])

    # find slug in text of anchor tag
    for slug in slug_attempts:
        pattern = re.compile(f"([^a-zA-Z])*{slug}(.|\n)*", flags=re.IGNORECASE)
        anchor_tags = soup.find_all("a", href=True)
        for a in anchor_tags:
            if pattern.match(a.text):
                return get_absolute_link(base_url, a["href"])
    return base_url


def add_line_up_links(events: List[dict]) -> List[dict]:
    """Add the url to the festival's line up page to festival dict.

    Args:
        events (List[dict]): all festivals

    Returns:
        List[dict]: all festivals with the line up link added

    """
    result = []
    total = len(events)
    last_percents = total * 0.97
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Start the load operations and mark each future with its URL
        future_to_url = {
            executor.submit(find_line_up_link_for_a_single_festival, event): event
            for event in events
        }
        for future in concurrent.futures.as_completed(future_to_url):
            event = future_to_url[future]
            try:
                line_up_url = future.result()
            except Exception as exc:
                print("%s generated an exception: %s" % (event["url"], exc))
            else:
                event["line_up_url"] = line_up_url
                result.append(event)
                if (current := len(result)) % 50 == 0 or current > last_percents:
                    print(
                        f"\r{float(current)/total*100:.1}%: {current}/{total}", end=""
                    )

    return result


def get_html_from_line_up_links(events: List[dict]) -> List[dict]:
    """Get html code of websites in parallel.

    Args:
        events (List[dict]): events. "line_up_url" contains the url to fetch.

    Returns:
        List[dict]: events with "scraped_line_up_html" added

    """
    line_up_urls = [e["line_up_url"] for e in events]

    results = asyncio.run(run_parallel_fetch(line_up_urls))
    results_dict = {}
    for url, soup in results:
        if soup is None or soup.body is None:
            results_dict[url] = None
            continue

        content = soup.body
        for script in content.find_all("script"):
            script.decompose()
        for svg in content.select("svg"):
            svg.decompose()
        for header in content.select("header"):
            header.decompose()
        results_dict[url] = content.decode()

    for event in events:
        event["scraped_line_up_html"] = results_dict[event["line_up_url"]]
    return events


def get_line_up_from_html(events: List[dict]) -> List[dict]:
    """Get line up as list from html code. Use Gemini API for this.

    Args:
        events (List[dict]): list of events. In "scraped_line_up_html" the html code is stored.

    Returns:
        List[dict]: Events with "line-up" added.

    """

    def single(event: dict):
        if event["scraped_line_up_html"] is None:
            event["line-up"] = []
            return event

        prompt = """List the bands mentioned in the following HTML code. Only return the bands you are sure about and list them in a python array. If you cannot find any bands return an empty array. Do not add country or city information about the bands, only list their names.\n\n"""
        try:
            response = gemini_generate(prompt + str(event["scraped_line_up_html"]))
        # to avoid recitation error: do not pass all the html code but only the text
        except ValueError:
            response = gemini_generate(
                prompt
                + BeautifulSoup(event["scraped_line_up_html"], "html.parser").text
            )

        try:
            bands = json.loads(response)
        except Exception:
            bands = []
        event["line-up"] = bands
        return event

    def single_wrapped_try_except(event: dict):
        """Wrap the single function in a try catch.

        This is needed to ensure that
        ThreadPoolExecuter will not have to catch an exception. This might lead to a
        deadlock:
        The ThreadPoolExecutor's thread pool catches the exception (it doesn't crash
        the entire program!) and stores it within the corresponding Future object.
        If the main thread later calls future.result() or future.exception() on the
        Future object (usually to get the result or to check for an exception), it
        will attempt to retrieve the stored exception. The result() and exception()
        methods internally call the internal lock and that's where you are stuck.

        Args:
            event (dict): event

        Returns:
            dict: event with "line-up" field added

        """
        try:
            return single(event)
        except Exception as e:
            print(f"Exception in single_wrapped_try_except for {event['name']}:", e)
            return event

    result = []
    total = len(events)
    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Start the load operations and mark each future with its URL
            future_to_url = {
                executor.submit(single_wrapped_try_except, event): event
                for event in events
            }
            for future in concurrent.futures.as_completed(future_to_url):
                event = future_to_url[future]
                try:
                    event = future.result(timeout=60 * 60 * 1.5)  # 1.5h for each item
                except concurrent.futures.TimeoutError:
                    print(
                        f"{event['name']} took too long: concurrent.futures.TimeoutError"
                    )
                except Exception as exc:
                    print("%s generated an exception: %s" % (event["url"], exc))
                else:
                    result.append(event)
                    if (current := len(result)) % 50 == 0 or current > total - 40:
                        print(
                            f"\r{float(current)/total*100:.1}%: {current}/{total}",
                            end="",
                        )
    except KeyboardInterrupt as e:
        print(e)
        print("shutting down executor")
        executor.shutdown(wait=False)
        print("Returning current list of festivals.")
    finally:
        return result


def clean_up(events: List[dict]) -> List[dict]:
    """Merge bands and line-up field and remove html code.

    Args:
        events (List[dict]): events

    Returns:
        List[dict]: cleaned up events

    """
    for event in events:
        lineup = event["line-up"] if "line-up" in event else []
        # in case gemini api returns [[<bands>]] instead of [<bands>]
        if len(lineup) and isinstance(lineup[0], list):
            lineup = lineup[0]
        unique_bands = list(
            set(s.upper().encode().decode("utf-8") for s in (lineup + event["bands"]))
        )
        if "" in unique_bands:
            unique_bands.remove("")
        event["bands"] = unique_bands
        del event["line-up"]
        del event["scraped_line_up_html"]
    return events


if __name__ == "__main__":
    load_dotenv(override=True)
    parser = argparse.ArgumentParser(description="Festival Data Pipeline")
    parser.add_argument(
        "--override",
        action="store_true",
        help="Override all steps and start from scratch",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("steps"),
        help="Directory to store JSON step files",
    )

    args = parser.parse_args()

    pipeline = [
        (
            f"{args.data_dir}/events_1_initial.json",
            lambda _: crawl_festivals_from_festival_ticker(),
        ),
        (f"{args.data_dir}/events_2_with_line_up_links.json", add_line_up_links),
        (f"{args.data_dir}/events_3_with_html_code.json", get_html_from_line_up_links),
        (f"{args.data_dir}/events_4_with_bands.json", get_line_up_from_html),
        (f"{args.data_dir}/events.json", clean_up),
    ]

    events = None
    start_index = 0
    if not args.override:
        for i in reversed(range(len(pipeline))):
            path = Path(pipeline[i][0])
            if path.exists():
                print(f"Resuming from: {path}")
                with open(path, "r") as f:
                    events = json.load(f)
                start_index = i + 1
                break

    for i in range(start_index, len(pipeline)):
        filename, func = pipeline[i]
        print(f"Running step {i + 1}: {func.__name__}...")

        start_time = datetime.now()
        events = func(events)
        duration = datetime.now() - start_time

        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        with open(filename, "w") as f:
            json.dump(events, f, indent=2)

        print(
            f"Step {i + 1} completed in {format_duration(duration.total_seconds())}\n"
        )
    exit(0)  # stop program and thereby kill all created threads
