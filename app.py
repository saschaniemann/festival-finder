"""Steamlit app to search festivals from a previously crawled list of festivals."""

import math
import streamlit as st
import pandas as pd
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


data = load_data()

# Preprocessing
data["start_date"] = pd.to_datetime(data["start_date"], format="%d.%m.%Y")
data["end_date"] = pd.to_datetime(data["end_date"], format="%d.%m.%Y")

# Extract unique filters
all_genres = sorted({g for sub in data["genres"] for g in sub})
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

if submitted:
    filtered = data.copy()
    if genres:
        filtered = filtered[
            filtered["genres"].apply(lambda gl: any(g in gl for g in genres))
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
            st.error(f"Fehler bei der Geokodierung: {e}")

    # Display results
    if filtered.empty:
        st.warning("Keine Festivals gefunden, die den Kriterien entsprechen.")
    else:
        for _, row in filtered.iterrows():
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
                genre if genre not in genres else f"**{genre}**"
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
