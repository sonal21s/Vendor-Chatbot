"""Runtime geocoding for the nearest-city fallback.

The CityCoord worksheet intentionally holds coordinates only for cities where
the company HAS vendors. When a user asks about a city with no vendors (and
therefore no row in CityCoord), we still need that city's coordinates to find
the closest vendor city. This module resolves such a city to (lat, lon) on the
fly via OpenStreetMap's free Nominatim API.

Kept deliberately defensive: any failure (network, no result, bad payload)
returns None so the caller treats it exactly like 'city unknown' and degrades
to the existing empty-result behaviour. Results are cached for the process
lifetime to respect Nominatim's usage policy and avoid repeat lookups.
"""

from functools import lru_cache

import requests

from src.utils import get_logger

log = get_logger(__name__)

_NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
# Nominatim's usage policy requires an identifying User-Agent.
_USER_AGENT = "VendorChatbot/1.0 (vendor-location-lookup)"
# All vendors are in India; biasing the query improves accuracy and
# disambiguates short / duplicated city names.
_DEFAULT_COUNTRY = "India"
_TIMEOUT_SECONDS = 5


@lru_cache(maxsize=512)
def geocode_city(place: str) -> tuple[float, float] | None:
    """Resolve a place name to (lat, lon) via Nominatim, or None on any miss.

    `place` may include state context, e.g. "Khopoli, Maharashtra". ", India"
    is appended automatically when no country is present.
    """
    if not place or not place.strip():
        return None
    query = place.strip()
    if _DEFAULT_COUNTRY.lower() not in query.lower():
        query = f"{query}, {_DEFAULT_COUNTRY}"

    try:
        resp = requests.get(
            _NOMINATIM_URL,
            params={"q": query, "format": "json", "limit": 1},
            headers={"User-Agent": _USER_AGENT},
            timeout=_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:  # network error, timeout, non-200, bad JSON
        log.warning("Geocoding failed for %r (%s); fallback skipped.", place, e)
        return None

    if not data:
        log.info("No geocoding result for %r.", query)
        return None
    try:
        lat = float(data[0]["lat"])
        lon = float(data[0]["lon"])
    except (KeyError, IndexError, TypeError, ValueError):
        log.warning("Unexpected geocoding payload for %r; fallback skipped.", query)
        return None

    log.info("Geocoded %r -> (%.4f, %.4f).", query, lat, lon)
    return (lat, lon)
