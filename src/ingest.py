import gspread
import pandas as pd
from settings import (
    GCP_CREDENTIALS,
    GOOGLE_SHEET_ID,
    GOOGLE_SHEET_NAME,
    CITY_COORDS_SHEET_NAME,
)
from src.utils import get_logger

log = get_logger(__name__)


def load_vendors_df() -> pd.DataFrame:
    log.info("Connecting to Google Sheets…")
    client = gspread.service_account_from_dict(GCP_CREDENTIALS)
    sheet = client.open_by_key(GOOGLE_SHEET_ID).worksheet(GOOGLE_SHEET_NAME)
    rows = sheet.get_all_records()
    df = pd.DataFrame(rows)
    df = df.fillna("").astype(str).apply(lambda c: c.str.strip())
    log.info("Loaded %d vendor rows with columns: %s", len(df), list(df.columns))
    return df


def load_city_coords() -> dict[str, tuple[float, float]]:
    """Load the CityCoord worksheet into a {city_lower: (lat, lon)} map.

    Powers the nearest-city fallback in query_executor. Kept deliberately
    tolerant: a missing/empty worksheet or unparseable row simply yields a
    smaller (or empty) map, which disables the fallback for the affected
    cities rather than breaking the app.
    """
    log.info("Loading city coordinates from worksheet '%s'…", CITY_COORDS_SHEET_NAME)
    try:
        client = gspread.service_account_from_dict(GCP_CREDENTIALS)
        sheet = client.open_by_key(GOOGLE_SHEET_ID).worksheet(CITY_COORDS_SHEET_NAME)
        rows = sheet.get_all_records()
    except Exception as e:  # worksheet absent, network error, auth, etc.
        log.warning("Could not load city coordinates (%s); nearest-city fallback disabled.", e)
        return {}

    coords: dict[str, tuple[float, float]] = {}
    for row in rows:
        city = str(row.get("City", "")).strip().lower()
        if not city:
            continue
        try:
            lat = float(row.get("Latitude"))
            lon = float(row.get("Longitude"))
        except (TypeError, ValueError):
            continue  # blank or non-numeric coordinate — skip this row
        coords[city] = (lat, lon)
    log.info("Loaded coordinates for %d cities.", len(coords))
    return coords
