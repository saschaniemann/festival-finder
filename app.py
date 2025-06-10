"""Steamlit app to search festivals from a previously crawled list of festivals."""

import math
import streamlit as st
import pandas as pd
import json
from datetime import datetime
from typing import List
from geopy.distance import geodesic

st.set_page_config(page_title="Festival finder", page_icon="favicon.png")
st.title("Festival Finder")


# Load JSON data
@st.cache_data
def load_data(path="events.json"):
    """Load data from json."""
    return pd.read_json(path)


def load_genre_map(path="cleaned_up_genres.json"):
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
) -> pd.DataFrame:
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
            # Geocode input location
            from geopy.geocoders import Nominatim

            geolocator = Nominatim(user_agent="festival_app")
            loc = geolocator.geocode(location_input)
            user_point = (loc.latitude, loc.longitude)
            # filter by distance
            filtered["distance_km"] = filtered.apply(
                lambda row: calc_distance(user_point, row), axis=1
            )
            filtered = filtered[filtered["distance_km"] <= max_distance]
        except Exception as e:
            st.error(
                f"Ort '{e}' konnte nicht gefunden werden. Bitte überprüfen Sie die Eingabe."
            )
    return filtered


def display_results(data: pd.DataFrame, genres: List[str], bands: List[str]):
    """Display the filtered results."""
    if data.empty:
        st.warning("Keine Festivals gefunden, die den Kriterien entsprechen.")
        return

    for _, row in data.iterrows():
        st.markdown(f"### {row['name']}")
        # Single-day or multi-day display
        if pd.isnull(row["end_date"]):
            date_str = row["start_date"].strftime("%d.%m.%Y")
        else:
            date_str = f"{row['start_date'].strftime('%d.%m.%Y')} - {row['end_date'].strftime('%d.%m.%Y')}"
        st.markdown(f"**Datum:** {date_str}")
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
    st.session_state.data = filter_data(
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
