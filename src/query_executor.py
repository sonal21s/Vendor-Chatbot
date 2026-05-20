import re
import pandas as pd
from rapidfuzz import process, fuzz
from settings import FUZZ_THRESHOLD
from src.utils import get_logger

log = get_logger(__name__)

# Any column whose name (lower) contains one of these keywords is treated
# as sensitive banking/payment data. Stripped from the response by default
# and only included when the user explicitly asks for a specific vendor.
_BANK_KEYWORDS = ("bank", "account", "ifsc", "upi", "branch")


def _is_bank_column(col_name: str) -> bool:
    lower = str(col_name).lower()
    return any(k in lower for k in _BANK_KEYWORDS)


_UNLIMITED_KEYWORDS = ("anywhere", "country", "india", "unlimited", "pan ", "nation", "all of india", "any state")
_UNLIMITED_KM = 9999.0


def parse_travel_km(value) -> float:
    """Convert a free-text Travel_Radius_KM cell into kilometres.

    Handles the shapes vendors actually enter:
        "Upto 250 km"             → 250
        "Local city within 50 km" → 50
        "25 kilometres"           → 25
        "Anywhere in the country" → 9999 (treated as effectively unlimited)
        ""  / "Local"             → NaN  (no extractable number)
    Strategy: keyword-check for unlimited phrasing first, otherwise pick the
    first integer/decimal in the string. Conservative when the text contains
    a range like "30 to 50 km" — picks the lower bound.
    """
    if value is None:
        return float("nan")
    text = str(value).strip().lower()
    if not text:
        return float("nan")
    if any(k in text for k in _UNLIMITED_KEYWORDS):
        return _UNLIMITED_KM
    match = re.search(r"\d+(?:\.\d+)?", text)
    if match:
        return float(match.group())
    return float("nan")


def _tier_priority(score) -> int:
    """Order tiers: Recommended → Good → New/Never Used → Risky."""
    s = pd.to_numeric(score, errors="coerce")
    if pd.isna(s) or s == 0:
        return 2          # New / Never Used (no track record)
    if s >= 0.8:
        return 0          # Recommended
    if s >= 0.5:
        return 1          # Good
    return 3              # Risky (known issues)


def _fuzzy_match_value(query_value: str, candidates: list[str]) -> str | None:
    if not query_value or not candidates:
        return None
    unique = list({c for c in candidates if c})
    match = process.extractOne(
        query_value, unique, scorer=fuzz.WRatio, score_cutoff=FUZZ_THRESHOLD
    )
    return match[0] if match else None


def _fuzzy_filter(df: pd.DataFrame, column: str, value: str) -> tuple[pd.DataFrame, str | None]:
    if not value or column not in df.columns:
        return df, None
    matched = _fuzzy_match_value(value, df[column].tolist())
    if matched is None:
        return df.iloc[0:0], None
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

    if slots.get("work_type") and "Work_Type" in filtered.columns:
        wt = slots["work_type"].strip().lower()
        for suffix in (" work", " job", " service", " services", " task"):
            if wt.endswith(suffix):
                wt = wt[: -len(suffix)].strip()
        col = filtered["Work_Type"].str.lower()
        contains_mask = col.str.contains(wt, na=False, regex=False)
        fuzzy_mask = col.apply(
            lambda v: bool(v) and fuzz.partial_ratio(wt, v) >= FUZZ_THRESHOLD
        )
        filtered = filtered[contains_mask | fuzzy_mask]
        applied["Work_Type~"] = wt

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

    if slots.get("min_travel_km") is not None and "Travel_Radius_KM" in filtered.columns:
        radii = filtered["Travel_Radius_KM"].apply(parse_travel_km)
        filtered = filtered[radii >= float(slots["min_travel_km"])]
        applied["min_travel_km"] = slots["min_travel_km"]

    # Vendor code is more specific than name — apply it first. If a code
    # is present, we don't also apply the name fuzzy match (would over-restrict).
    code_value = slots.get("vendor_code")
    name_value = slots.get("vendor_name")

    if code_value and "Vendor_Code" in filtered.columns:
        code = str(code_value).strip().lower()
        col = filtered["Vendor_Code"].fillna("").astype(str).str.strip().str.lower()
        filtered = filtered[col == code]
        applied["Vendor_Code"] = code_value
    elif name_value and "Vendor_Name" in filtered.columns:
        query_norm = str(name_value).strip().lower()
        # Build lower→original map so we can fuzz-match case-insensitively
        # while still filtering against the original-cased column.
        originals = filtered["Vendor_Name"].fillna("").astype(str).tolist()
        lower_to_originals: dict[str, list[str]] = {}
        for orig in originals:
            lower_to_originals.setdefault(orig.strip().lower(), []).append(orig)

        matches = process.extract(
            query_norm,
            list(lower_to_originals.keys()),
            scorer=fuzz.token_set_ratio,
            score_cutoff=FUZZ_THRESHOLD,
            limit=10,
        )
        matched_originals: list[str] = []
        for matched_lower, _, _ in matches:
            matched_originals.extend(lower_to_originals[matched_lower])

        # Fallback: if the user's "name" had no spaces (likely a code that
        # the slot filler misclassified), also try exact code match.
        if not matched_originals and " " not in str(name_value).strip() \
                and "Vendor_Code" in filtered.columns:
            code = str(name_value).strip().lower()
            code_col = filtered["Vendor_Code"].fillna("").astype(str).str.strip().str.lower()
            code_match = filtered[code_col == code]
            if len(code_match) > 0:
                filtered = code_match
                applied["Vendor_Code"] = name_value
            else:
                filtered = filtered.iloc[0:0]
                applied["Vendor_Name~"] = name_value
        else:
            filtered = filtered[filtered["Vendor_Name"].isin(matched_originals)]
            applied["Vendor_Name~"] = name_value

    # Rank by tier-then-score: Recommended → Good → New/Never Used → Risky.
    # Within each tier, sort by score descending.
    if "overall_score" in filtered.columns and len(filtered) > 1:
        scores = pd.to_numeric(filtered["overall_score"], errors="coerce")
        priorities = scores.apply(_tier_priority)
        filtered = (
            filtered.assign(_pri=priorities, _score=scores)
            .sort_values(["_pri", "_score"], ascending=[True, False], na_position="last")
            .drop(columns=["_pri", "_score"])
        )

    total_matches = len(filtered)

    # Apply user-requested top-N limit AFTER sorting.
    limit = slots.get("limit")
    if isinstance(limit, int) and limit > 0:
        filtered = filtered.head(limit)
        applied["limit"] = limit

    # ------------------------------------------------------------------
    # Bank-detail gating. The LLM NEVER sees bank columns unless we
    # explicitly include them here, and we only include them when:
    #   (a) the user explicitly asked for bank details, AND
    #   (b) the result is unambiguously a single specific vendor.
    # Otherwise: strip bank columns and signal that clarification is
    # needed so the response generator can ask the user to specify.
    # ------------------------------------------------------------------
    bank_columns = [c for c in filtered.columns if _is_bank_column(c)]
    bank_included = False
    bank_clarification_needed = False
    intent = slots.get("intent", "list")

    if slots.get("requests_bank_details"):
        # Allowed only for single-vendor lookups.
        is_single_vendor = intent in ("lookup", "details") and len(filtered) == 1
        if is_single_vendor:
            bank_included = True  # keep bank columns visible
        else:
            bank_clarification_needed = True

    if not bank_included and bank_columns:
        filtered = filtered.drop(columns=bank_columns, errors="ignore")

    return {
        "rows": filtered,
        "count": len(filtered),
        "total_matches": total_matches,
        "applied_filters": applied,
        "intent": intent,
        "bank_included": bank_included,
        "bank_clarification_needed": bank_clarification_needed,
    }
