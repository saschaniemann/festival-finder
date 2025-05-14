"""Utility functions that can be used throughout the entire project."""

import requests
import os
from typing import Tuple, Optional
from requests.structures import CaseInsensitiveDict
import urllib.parse


def get_lat_long(location: str) -> Tuple[Optional[float], Optional[float]]:
    """Get latitude and longitude from address string. If not found, `None, None` is returned.

    Args:
        location (str): address

    Returns:
        Tuple[Optional[float], Optional[float]]: lat, long as float or None, None

    """
    encoded_address = urllib.parse.quote(location)
    url = f"https://api.geoapify.com/v1/geocode/search?text={encoded_address}&apiKey={os.getenv('GEOAPIFY_API_KEY')}"

    headers = CaseInsensitiveDict()
    headers["Accept"] = "application/json"

    try:
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()

        data = resp.json()
        features = data.get("features")

        if features:
            coords = features[0]["geometry"]["coordinates"]
            longitude, latitude = coords[0], coords[1]
            return latitude, longitude
        else:
            print(
                f"[GEOCODER]: No coordinates found for the given address. ({location})"
            )
            return None, None
    except requests.RequestException as e:
        print(f"[GEOCODER]: Request failed: {e}")
        return None, None
