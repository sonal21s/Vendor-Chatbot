import gspread
import pandas as pd
from settings import GCP_CREDENTIALS, GOOGLE_SHEET_ID, GOOGLE_SHEET_NAME
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
