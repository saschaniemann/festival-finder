"""Crawl festival info that is later searchable and filterable on a frontend."""

import requests
import json
import re
from bs4 import BeautifulSoup
from tqdm import tqdm
from typing import List
from urllib.parse import urljoin, urlparse


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

        # if 'show more genres' was available: genres='<short-list>,... mehr\n<long-list>', only keep <long-list>
        show_more_text = ",... mehr\n"
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

        bands = website_bands_rows[1].text.strip()
        if not bands.startswith("Bands:"):
            return []

        bands = bands.split("\n")[1]
        bands = bands.split(", ")
        bands[0] = bands[0].strip()  # remove leading whitespaces
        # if 'und weitere': remove it
        if bands[-1].endswith("\rund weitere"):
            bands[-1] = bands[-1].split("\r")[0]

        return bands

    response = get_request(url)
    soup = BeautifulSoup(response.content, "html.parser")

    main_info = soup.body.find_next("table")
    main_info = main_info.find_next("td").find_next("table").find_next("td")

    start_date, end_date = extract_date(main_info)
    genres = extract_genres(main_info)
    url = extract_festival_url(main_info)
    bands = extract_bands(main_info)

    return {
        "start_date": start_date,
        "end_date": end_date,
        "genres": genres,
        "url": url,
        "bands": bands,
    }


def crawl_festivals_from_festival_ticker() -> List[dict]:
    """Crawl all festivals from festivalticker.de with its important information.

    Returns:
        List[dict]: list of festivals

    """
    events = get_festival_list()

    events_updated = []
    for event in tqdm(events):
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

    slug_attempts = ["line-up", "bands", "lineup"]
    base_url = festival["url"]

    # try urls as good-luck-guesses
    for slug in slug_attempts:
        url = urljoin(base_url, slug)
        response = requests.get(url)
        if response.status_code == 200:
            return url

    response = get_request(base_url)
    soup = BeautifulSoup(response.content, "html.parser")

    # find slug in href of anchor tags that match the slug
    for slug in slug_attempts:
        pattern = re.compile(f"(.|\n)*{slug}(.|\n)*", flags=re.IGNORECASE)
        anchor_tags = soup.find_all("a", href=True, string=pattern)
        for a in anchor_tags:
            if pattern.match(a["href"]):
                return get_absolute_link(base_url, a["href"])

    # find slug in text of anchor tag
    for slug in slug_attempts:
        pattern = re.compile(f"(.|\n)*{slug}(.|\n)*", flags=re.IGNORECASE)
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
    for event in events:
        line_up_url = find_line_up_link_for_a_single_festival(event)
        event["line_up_url"] = line_up_url
        result.append(event)
    return result


if __name__ == "__main__":
    events = crawl_festivals_from_festival_ticker()
    events = add_line_up_links(events)

    with open("events.json", "w") as f:
        json.dump(events, f)
