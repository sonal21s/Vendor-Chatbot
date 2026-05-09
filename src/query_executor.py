import pandas as pd
from rapidfuzz import process, fuzz
from settings import FUZZ_THRESHOLD
from src.utils import get_logger

log = get_logger(__name__)


def _fuzzy_match_value(query_value: str, candidates: list[str]) -> str | None:
    """Return the best-matching canonical value from candidates, or None."""
    if not query_value or not candidates:
        return None
    unique = list({c for c in candidates if c})
    match = process.extractOne(
        query_value, unique, scorer=fuzz.WRatio, score_cutoff=FUZZ_THRESHOLD
    )
    return match[0] if match else None


def _fuzzy_filter(df: pd.DataFrame, column: str, value: str) -> tuple[pd.DataFrame, str | None]:
    """Filter df where column ≈ value (fuzzy). Returns (filtered_df, matched_value)."""
    if not value or column not in df.columns:
        return df, None
    matched = _fuzzy_match_value(value, df[column].tolist())
    if matched is None:
        return df.iloc[0:0], None  # empty
    return df[df[column].str.lower() == matched.lower()], matched


def execute(df: pd.DataFrame, slots: dict) -> dict:
    """Apply slot filters to df. Returns dict with filtered rows + applied filters."""
    filtered = df.copy()
    applied = {}

    if slots.get("state"):
        filtered, matched = _fuzzy_filter(filtered, "State", slots["state"])
        if matched:
            applied["State"] = matched

    if slots.get("city"):
        filtered, matched = _fuzzy_filter(filtered, "City", slots["city"])
        if matched:
            applied["City"] = matched

    if slots.get("work_type"):
        filtered, matched = _fuzzy_filter(filtered, "Work_Type", slots["work_type"])
        if matched:
            applied["Work_Type"] = matched

    if slots.get("recommendation") and "Recommendation" in filtered.columns:
        tag = slots["recommendation"]
        filtered = filtered[filtered["Recommendation"].str.lower() == tag.lower()]
        applied["Recommendation"] = tag

    if slots.get("exclude_risky") and "Recommendation" in filtered.columns:
        filtered = filtered[filtered["Recommendation"].str.lower() != "risky"]
        applied["Recommendation"] = "not Risky"

    if slots.get("min_rating") is not None and "Quality_Rating" in filtered.columns:
        ratings = pd.to_numeric(filtered["Quality_Rating"], errors="coerce")
        filtered = filtered[ratings >= float(slots["min_rating"])]
        applied["min_rating"] = slots["min_rating"]

    if slots.get("vendor_name"):
        # Fuzzy match on vendor name — partial-match friendly
        names = filtered["Vendor_Name"].tolist() if "Vendor_Name" in filtered.columns else []
        if names:
            matches = process.extract(
                slots["vendor_name"], names,
                scorer=fuzz.token_set_ratio,
                score_cutoff=FUZZ_THRESHOLD, limit=10,
            )
            matched_names = [m[0] for m in matches]
            filtered = filtered[filtered["Vendor_Name"].isin(matched_names)]
            applied["Vendor_Name~"] = slots["vendor_name"]

    return {
        "rows": filtered,
        "count": len(filtered),
        "applied_filters": applied,
        "intent": slots.get("intent", "list"),
    }
