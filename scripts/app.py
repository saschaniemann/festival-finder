"""Steamlit app to search festivals from a previously crawled list of festivals."""

import math
from types import SimpleNamespace
import streamlit as st
import pandas as pd
import json
from datetime import datetime
from typing import List, Tuple, Any
from geopy.distance import geodesic
import pydeck as pdk
from utils import get_lat_long
from dotenv import load_dotenv

st.set_page_config(page_title="Festival finder", page_icon="assets/favicon.png")
st.title("Festival Finder")


@st.cache_resource(ttl="1h", show_spinner=False)
def load_env():
    """Load environment variables from .env file."""
    load_dotenv()


# Load JSON data
@st.cache_data
def load_data(path="data/events.json"):
    """Load data from json."""
    return pd.read_json(path)


@st.cache_data
def load_genre_map(path="data/cleaned_up_genres.json"):
    """Load genre map from json."""
    with open(path, "r") as f:
        return json.load(f)


@st.cache_data(ttl="1h", show_spinner=False)
def filter_data(
    data: pd.DataFrame,
    genres: List[str],
    selected_months: List[int],
    bands: List[str],
    location_input: str,
    max_distance: int,
) -> Tuple[pd.DataFrame, Any]:
    """Filter the data based on user input."""
    filtered = data.copy()
    if genres:
        filtered = filtered[
            filtered["genres"].apply(
                lambda gl: any(
                    any(genre in genre_map[g] for genre in genres) for g in gl
                )
            )
        ]
    if selected_months:

        def in_selected_months(row) -> bool:
            """Check whether the passed row is one of the selected months."""
            if row["start_date"].month in selected_months:
                return True
            if pd.notnull(row["end_date"]) and row["end_date"].month in selected_months:
                return True
            return False

        filtered = filtered[filtered.apply(in_selected_months, axis=1)]
    if bands:
        filtered = filtered[
            filtered["bands"].apply(lambda bl: any(b in bl for b in bands))
        ]

    def calc_distance(user_point, festival):
        """Calculate distance in km between user_point and festival location."""
        if not "location" in festival:
            return math.inf
        location = festival["location"]
        if not "latitude" in location or not "longitude" in location:
            return math.inf
        return geodesic(user_point, (location["latitude"], location["longitude"])).km

    if location_input:
        try:
            # geolocator = Nominatim(user_agent="festival_app")
            # loc = geolocator.geocode(location_input)
            latitude, longitude = get_lat_long(location_input)
            if latitude is None or longitude is None:
                raise ValueError("Invalid location input")
            loc = SimpleNamespace(latitude=latitude, longitude=longitude)
            user_point = (loc.latitude, loc.longitude)
            # filter by distance
            filtered["distance_km"] = filtered.apply(
                lambda row: calc_distance(user_point, row), axis=1
            )
            filtered = filtered[filtered["distance_km"] <= max_distance]
            return filtered, loc
        except Exception as e:
            print(f"Error processing location input '{location_input}': {e}")
            st.error(
                f"Ort '{location_input}' konnte nicht gefunden werden. Bitte überprüfen Sie die Eingabe."
            )
    return filtered, None


def display_map(data: pd.DataFrame, loc: Any) -> None:
    """Display the festival locations on a map.

    This function creates a map using pydeck to visualize festival locations
    and the user's location. It uses latitude and longitude from the data
    and adds markers for each festival and the user's location.

    Args:
        data (pd.DataFrame): festival data
        loc (Any): location containing latitude and longitude of the user

    """
    if data is None or data.empty or loc is None:
        return

    # Prepare data for map
    map_data = pd.DataFrame(
        {
            "lat": [
                row["location"]["latitude"]
                for _, row in data.iterrows()
                if "location" in row and "latitude" in row["location"]
            ],
            "lon": [
                row["location"]["longitude"]
                for _, row in data.iterrows()
                if "location" in row and "longitude" in row["location"]
            ],
            "info": [
                row["name"]
                for _, row in data.iterrows()
                if "location" in row and "latitude" in row["location"]
            ],
            "is_user": [False] * len(data),
        }
    )

    # Add user location
    map_data = pd.concat(
        [
            map_data,
            pd.DataFrame(
                {
                    "lat": [loc.latitude],
                    "lon": [loc.longitude],
                    "info": ["Your Location"],
                    "is_user": [True],
                }
            ),
        ]
    )

    map_data["icon_data"] = [
        {
            "url": "https://img.icons8.com/ios-filled/50/FA5252/marker-a.png"
            if row
            else "https://img.icons8.com/ios-filled/50/4a90e2/full-stop--v1.png",
            "width": 64 if row else 32,
            "height": 64 if row else 32,
            "anchorY": 64 if row else 32,
        }
        for row in map_data["is_user"]
    ]

    icon_layer = pdk.Layer(
        type="IconLayer",
        data=map_data,
        get_icon="icon_data",
        get_size=4,
        size_scale=10,
        get_position="[lon, lat]",
        pickable=True,
    )

    tooltip = {"html": "<b>{info}</b>", "style": {"color": "white"}}

    st.pydeck_chart(
        pdk.Deck(
            map_style="light",
            initial_view_state=pdk.ViewState(
                latitude=loc.latitude, longitude=loc.longitude, zoom=5, pitch=0
            ),
            layers=[icon_layer],
            tooltip=tooltip,
        )
    )


def display_results(data: pd.DataFrame, genres: List[str], bands: List[str]):
    """Display the filtered results."""
    if data.empty:
        st.warning("Keine Festivals gefunden, die den Kriterien entsprechen.")
        return

    if "distance_km" in data.columns and "location" in st.session_state:
        with st.expander("Festival-Standorte auf Karte anzeigen"):
            location_input = st.session_state.location
            display_map(data, location_input)

    for _, row in data.iterrows():
        st.markdown(f"### {row['name']}")
        # Single-day or multi-day display
        if pd.isnull(row["end_date"]):
            date_str = row["start_date"].strftime("%d.%m.%Y")
        else:
            date_str = f"{row['start_date'].strftime('%d.%m.%Y')} - {row['end_date'].strftime('%d.%m.%Y')}"
        st.markdown(f"**Datum:** {date_str}")
        st.markdown(f"**Ort:** {row['location']['city']}, {row['location']['country']}")
        if "distance_km" in row:
            st.markdown(f"**Entfernung:** {row['distance_km']:.1f} km")
        genres_formatted = [
            genre if not any(g in genre_map[genre] for g in genres) else f"**{genre}**"
            for genre in row["genres"]
        ]
        st.markdown(f"**Genres:** {', '.join(genres_formatted)}")
        bands_formatted = [
            band if band not in bands else f"**{band}**" for band in row["bands"]
        ]
        st.markdown(f"**Bands:** {', '.join(bands_formatted)}")
        st.markdown(
            f"[Festival-Website]({row['url']}) | [Line-up]({row['line_up_url']}) | [Festivalticker]({row['festival_ticker_url']})"
        )
        st.markdown("---")


load_env()
data = load_data()
genre_map = load_genre_map()

# Preprocessing
data["start_date"] = pd.to_datetime(data["start_date"], format="%d.%m.%Y")
data["end_date"] = pd.to_datetime(data["end_date"], format="%d.%m.%Y")

# Extract unique filters
all_genres = set(genre_map.values())
all_genres = all_genres - set(["...", "Unknown"])
all_genres = sorted(all_genres)
all_bands = sorted({b for sub in data["bands"] for b in sub})
month_numbers = list(range(1, 13))  # 1 to 12
month_names = {i: datetime(2025, i, 1).strftime("%B") for i in month_numbers}

with st.form("filter_form"):
    genres = st.multiselect("Genres", options=all_genres)
    months = [None] + month_numbers
    selected_months = st.multiselect(
        "Monate (optional)", options=month_numbers, format_func=lambda x: month_names[x]
    )
    bands = st.multiselect("Bands", options=all_bands)
    location_input = st.text_input("Ort oder PLZ (z.B. 10115 Berlin, DE)")
    max_distance = st.slider(
        "Maximale Entfernung (km)", min_value=0, max_value=1000, value=100
    )
    submitted = st.form_submit_button("Filter anwenden")

if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame()

if submitted:
    st.session_state.data, st.session_state.location = filter_data(
        data, genres, selected_months, bands, location_input, max_distance
    )

if not st.session_state.data.empty:
    sort_by_options = {
        "Datum": "start_date",
        "Name": "name",
    }
    if "distance_km" in st.session_state.data.columns:
        sort_by_options["Entfernung"] = "distance_km"

    sort_by = st.selectbox("↑ Sortieren nach", options=sort_by_options.keys(), index=0)
    st.session_state.data = st.session_state.data.sort_values(
        by=sort_by_options.get(sort_by, "Datum"), ascending=True
    )
    display_results(st.session_state.data, genres, bands)
