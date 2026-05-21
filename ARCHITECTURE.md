# Vendor Chatbot — Architecture

A natural-language interface over a vendor database stored in Google Sheets. A service-procurement team asks questions in plain English ("how many recommended vendors in Pune?", "bank details for Vijay Rajput", "top 5 willing to travel 50km from Mumbai") and gets exact, structured answers with contact info.

## Design philosophy

The app deliberately avoids a RAG / vector-store architecture. The data is a small (~500-row) structured table — every question maps to a deterministic SQL-style operation (filter, count, sort, lookup). Embeddings would only add latency and inaccuracy here.

Instead, the LLM is used for the **two things it's actually good at**:

1. **Slot filling** — turning a messy natural-language query into a JSON object of canonical filter parameters (state, city, work_type, recommendation tier, distance, etc.).
2. **Answer phrasing** — composing a short, professional intro sentence based on the result shape.

Everything in between — filtering, counting, sorting, bank-detail gating — is **pure pandas + rapidfuzz**, deterministic and exact.

This means counts are always correct (no top-k truncation), typos are handled predictably, and the LLM cannot leak sensitive fields (bank details are stripped before the LLM ever sees them).

## End-to-end request flow

```
[ User types in chat ]
        │
        ▼
┌──────────────────────────────────────────────────────────────────────┐
│ app.py — Streamlit chat loop                                         │
│   • Password gate (st.secrets["auth"]["password"])                   │
│   • Loads cached DataFrame from Google Sheets                        │
│   • Captures user message, builds short conversation history         │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│ src/slot_filler.py · extract_slots(query, history)                  │
│ LLM (Groq LLaMA 3.3 70B) returns JSON:                               │
│   { intent, state, city, work_type, vendor_name, vendor_code,        │
│     recommendation, exclude_risky, min_rating, min_travel_km,        │
│     limit, requests_bank_details }                                   │
│ • Normalises state names ("andhra" → "Andhra Pradesh")               │
│ • Distinguishes vendor codes (no spaces) from names (whitespace)     │
│ • Captures distance phrasing ("travel 40km") → min_travel_km         │
│ • Detects bank-detail requests; treats fresh queries as independent  │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│ src/query_executor.py · execute(df, slots)                          │
│ Pure pandas + rapidfuzz. No LLM calls.                               │
│ • Fuzzy match State/City/Work_Type (rapidfuzz WRatio ≥ 80)           │
│ • Exact case-insensitive match on Vendor_Code                        │
│ • Case-insensitive token-set match on Vendor_Name                    │
│ • Numeric filters: min_rating, min_travel_km                         │
│ • Tier-aware sort: Recommended → Good → New/Never → Risky            │
│ • Top-N limit applied after sort                                     │
│ • BANK GATE: strip bank columns unless requests_bank_details=True    │
│   AND the result is a single uniquely identified vendor              │
│ • Returns: { rows, count, total_matches, applied_filters,            │
│              intent, bank_included, bank_clarification_needed }      │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│ src/response_generator.py · generate_response(query, result, ...)   │
│ LLM call #2. Writes ONLY the intro sentence — never the cards.       │
│ • Acknowledges count or top-N framing                                │
│ • Asks for clarification when bank query is ambiguous                │
│ • No emojis, no marketing prose                                      │
│ • Falls back to deterministic formatter on LLM error                 │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│ app.py · render_vendor_card(row)                                    │
│ For each matched row, render st.expander:                            │
│ • Collapsed: "Vendor_Name — Recommendation_Tag"                      │
│ • Expanded: Vendor Code, Location, Work Type, Quality Rating,        │
│   Travel Radius, Primary Contact                                     │
│ • Bank details section (only if executor authorised exposure)        │
└──────────────────────────────────────────────────────────────────────┘
```

## Directory layout

```
Vendorchatbot/
├── app.py                         Streamlit UI + orchestration + card rendering
├── settings.py                    Loads secrets, exposes typed constants
├── requirements.txt
├── README.md
├── ARCHITECTURE.md                This document
│
├── .streamlit/
│   ├── secrets.toml               Real credentials (gitignored)
│   └── secrets.toml.example       Template
│
├── src/
│   ├── __init__.py
│   ├── ingest.py                  Google Sheets → pandas DataFrame
│   ├── slot_filler.py             LLM call #1 — query → structured filters
│   ├── query_executor.py          pandas + rapidfuzz filtering + bank gate
│   ├── response_generator.py      LLM call #2 — result → intro sentence
│   ├── answer_formatter.py        Deterministic formatter (fallback only)
│   └── utils.py                   Logger + Streamlit caching helpers
│
└── tests/
    ├── test_query_executor.py     Filter, sort, gate, parser tests
    └── test_answer_formatter.py   Formatter / fallback tests
```

## Module reference

### `settings.py`

Loads runtime configuration from `st.secrets` and exposes module-level constants. Centralising this means all secret access goes through one file, and the rest of the codebase doesn't touch `st.secrets` directly.

Exposes:
- `GROQ_API_KEY` — Groq API key
- `GOOGLE_SHEET_ID`, `GOOGLE_SHEET_NAME` — source sheet identifiers
- `GCP_CREDENTIALS` — service-account dict for gspread
- `LLM_MODEL` — `"llama-3.3-70b-versatile"`
- `FUZZ_THRESHOLD` — rapidfuzz cutoff (default 80, range 0–100)

### `app.py` — Streamlit entry point

The main entry point. Responsibilities: authenticate user, load cached data, manage chat state, orchestrate the slot_filler → executor → response_generator pipeline, and render the result as expandable Streamlit cards.

| Function | Purpose |
|---|---|
| `check_password()` | Login gate. Renders a form, compares against `st.secrets["auth"]["password"]`, sets `st.session_state.authenticated`. Called before any other app logic — uses `st.stop()` to halt rendering for unauthenticated sessions. |
| `_clean_bank_label(col)` | Strips `(Company)` / `(Individual)` suffixes from a bank column name so the row label reads naturally. |
| `_is_bank_col(col)` | Returns `True` if the column name contains any of: `bank`, `account`, `ifsc`, `upi`, `branch`. |
| `_classify_owner(col)` | Maps a bank column name to one of `"company"`, `"individual"`, `"other"` based on its suffix. |
| `_collect_bank_by_owner(row)` | Groups all bank fields in a row into `{ company: {label: val}, individual: {label: val}, other: {…} }`. |
| `_render_bank_section(title, fields)` | Renders one labeled bank section. Always shows the three required fields (Bank Account Holder Name, Bank Account Number, IFSC Code) — uses `<blank>` placeholder for any missing value so the user can see what's not in the dataset. |
| `render_vendor_card(row, expanded, key_suffix)` | Core UI primitive. Renders one vendor as `st.expander`. Collapsed shows `Vendor_Name — Recommendation`. Expanded shows Vendor Code, Location, Work Type, Quality Rating, Travel Radius, Primary Contact, and bank sections if present. |
| `render_assistant_message(msg)` | Renders a stored assistant message from session state. Used to redraw past turns when Streamlit reruns the script. |
| Main chat loop | Reads `st.chat_input`, runs the three-stage pipeline, renders the LLM intro followed by cards, and stores the assistant message in `st.session_state.messages` as a structured dict (intro + rows + expand_first flag) so historical turns re-render correctly. |

Session-state format for assistant messages:

```python
{
    "role": "assistant",
    "content": {
        "intro": "<LLM intro string>",
        "rows": [<dict per matched vendor>],
        "expand_first": True,  # auto-expand only for single-vendor lookups
        "turn": <int>,
    }
}
```

### `src/ingest.py`

Single function: pull the sheet, normalise it, return a DataFrame.

| Function | Purpose |
|---|---|
| `load_vendors_df()` | Authenticates with gspread using `GCP_CREDENTIALS`, opens the configured sheet, calls `get_all_records()`, fills NaN with empty strings, strips whitespace from every cell, and returns a `pd.DataFrame`. Called once per session via the `@st.cache_resource` decorator in `utils.cached_dataframe`. |

### `src/slot_filler.py` — LLM call #1

Translates the user's natural-language query into a structured JSON object.

| Function | Purpose |
|---|---|
| `_default_slots()` | Returns a dict with every slot set to its default value. Used to backfill missing fields when the LLM returns partial JSON, and as the fallback on parse error. |
| `extract_slots(query, history)` | The main entry point. Sends the system prompt + (optional) last 2 conversation turns + the current query to Groq with `response_format={"type": "json_object"}` and `temperature=0`. Returns a dict matching the schema below. |

Slot schema:

```python
{
    "intent":                "count" | "list" | "lookup" | "details",
    "state":                 str | None,    # canonical full name, LLM-normalised
    "city":                  str | None,    # raw user spelling
    "work_type":             str | None,    # raw user spelling
    "vendor_name":           str | None,    # name with whitespace
    "vendor_code":           str | None,    # identifier with no whitespace
    "recommendation":        "Recommended" | "Good" | "Risky" | "New / Never Used" | None,
    "exclude_risky":         bool,
    "min_rating":            float | None,  # numeric Quality_Rating threshold
    "min_travel_km":         float | None,  # numeric Travel_Radius_KM threshold
    "limit":                 int | None,    # "top N"
    "requests_bank_details": bool,
}
```

Key prompt rules:
- **State normalisation** is the LLM's responsibility — it expands "andhra"/"MH"/"tamil" to the canonical full name. (Cities/names stay raw because the LLM doesn't know what's actually in the database.)
- **Vendor code vs name**: a token with no whitespace is treated as a code (covers `ArunJayprakaMumbaiVend`, `VIJAYBILAVN`, `ABCD-001`). A multi-word phrase is treated as a name.
- **History handling**: fresh queries are treated independently. Filters from earlier turns are only carried over for explicit referential follow-ups ("show me the recommended ones", "now filter by Mumbai"). There's also an explicit exception so that when the assistant asks "which vendor?" and the user replies with a name, the system maintains the bank-details request from the prior turn.

### `src/query_executor.py` — deterministic filtering

The most-tested module. All filtering, counting, sorting, and the bank-detail gate live here. No LLM calls.

| Function | Purpose |
|---|---|
| `parse_travel_km(value)` | Converts a free-text `Travel_Radius_KM` cell to a float. Returns the first integer in the string; phrases like "Anywhere in the country" / "Pan India" map to 9999 (effectively unlimited); empty / unparseable values return NaN. |
| `_tier_priority(score)` | Returns 0/1/2/3 for tier-aware sort. Priority 0 = Recommended (≥0.8), 1 = Good (≥0.5), 2 = New/Never Used (=0 or missing), 3 = Risky (>0, <0.5). |
| `_fuzzy_match_value(query, candidates)` | Picks the single best-matching canonical value using `rapidfuzz.process.extractOne` with `fuzz.WRatio` and the configured cutoff. Returns `None` if nothing scores high enough. |
| `_fuzzy_filter(df, column, value)` | Applies `_fuzzy_match_value` against a column and returns `(filtered_df, matched_value)`. Used for State and City. |
| `_is_bank_column(col_name)` | Returns `True` if the column name contains a bank-related keyword. |
| `execute(df, slots)` | Main entry. Applies filters in order: State → City → Work_Type → Recommendation → exclude_risky → min_rating → min_travel_km → Vendor_Code / Vendor_Name → tier-sort → top-N limit → bank gate. Returns a result dict. |

Key behaviours within `execute`:

- **Vendor code lookup** strips whitespace and lowercases both sides. Takes precedence over `vendor_name` if both are present.
- **Vendor name lookup** is case-insensitive via a `lower → originals` map: candidates are lowercased before `fuzz.token_set_ratio`, then results are mapped back to the original-cased names for the row filter. (rapidfuzz's `token_set_ratio` is case-sensitive by default — this is the fix.)
- **Self-correcting fallback**: if `vendor_name` is set but contains no whitespace and the name fuzzy match returns nothing, the executor automatically retries the value as a `Vendor_Code`. This makes the system resilient to the LLM occasionally misclassifying a code as a name.
- **Work_Type matching** is the most lenient — strips trailing filler words (`work`, `job`, `service`), then uses `str.contains` plus `partial_ratio` so multi-value cells like *"Interior Audit, Carpentry"* still match queries for "interior audit".
- **Sort order** is tier-then-score-desc: Recommended first (highest scores within tier), then Good, then New/Never Used (untested), then Risky (known issues last).
- **Bank-detail gate**: bank columns are kept in the result **only when** `requests_bank_details=True` **and** the intent is `lookup`/`details` **and** exactly one row remains. Otherwise bank columns are dropped from the DataFrame entirely, and `bank_clarification_needed=True` is set so the response generator can ask the user to specify which vendor. **The LLM never sees bank columns unless this gate explicitly passes them through.**

Result dict shape:

```python
{
    "rows":            pd.DataFrame,  # filtered, sorted, bank-stripped (or not)
    "count":           int,           # len(rows) after limit
    "total_matches":   int,           # len(rows) before limit
    "applied_filters": dict,          # canonical filter values used
    "intent":          str,
    "bank_included":   bool,
    "bank_clarification_needed": bool,
}
```

### `src/response_generator.py` — LLM call #2

Composes a short professional intro sentence. **Does not render vendor cards** — those are rendered by `app.py` as Streamlit components.

| Function | Purpose |
|---|---|
| `_build_context(result)` | Trims `result["rows"]` to `MAX_ROWS_IN_CONTEXT` (40), serialises to records, builds the JSON payload the LLM receives. |
| `generate_response(query, result, history)` | Sends the system prompt + history + user payload to Groq with `temperature=0`. Returns the intro string. On any exception, falls back to `answer_formatter.format_result(result)`. |

Prompt design highlights:
- **Strict separation** — the prompt repeatedly tells the LLM it does **not** filter, sort, count, or render cards. Its sole job is the intro.
- **No emojis, no marketing prose, no apologies** — professional, operational tone.
- **No hallucination** — must use only fields from the JSON; if a field is empty, omit it.
- **Bank policy** — if `bank_clarification_needed=True`, the entire reply is a polite request for the exact vendor name or code. No other content.
- **Tier framing** — describes the 4-tier system; instructs the LLM to present "New / Never Used" neutrally, not as a quality judgement.

### `src/answer_formatter.py` — deterministic fallback

Used when the LLM call fails. Mirrors what the LLM would produce, in plain markdown. Also the source of truth for the bank-section grouping logic that `app.py` uses.

| Function | Purpose |
|---|---|
| `_rec_label(value)` | Maps the raw recommendation string to the canonical label (handles case variations). |
| `_contact_line(row)` | Builds the `Primary Contact: … · Email: …` line. |
| `_filter_summary(applied)` | Formats `applied_filters` into a human-readable string like `State=Maharashtra, City=Mumbai`. |
| `_is_bank_field(col_name)` | Same keyword check as the executor. |
| `_classify_bank_field(col_name)` | Returns `"company"` / `"individual"` / `"other"` based on the suffix. |
| `_clean_bank_label(col_name)` | Strips suffix and replaces underscores with spaces. |
| `format_count(result)` | One-line bold count message. |
| `_vendor_block(r)` | Single vendor as a plain-text block (used by the fallback list). |
| `format_list(result, max_rows)` | Joined vendor blocks for `intent="list"` results. |
| `format_lookup(result)` | Single profile (or multi-match list) with the structured bank sections. Always renders the three required bank fields with `<blank>` placeholders for missing values. |
| `format_bank_clarification(result)` | Polite request for the exact vendor name / code, with candidate list when multiple matches exist. |
| `format_result(result)` | Top-level dispatcher — routes to the right formatter based on intent and the bank-clarification flag. |

### `src/utils.py`

Tiny utilities shared across modules.

| Function | Purpose |
|---|---|
| `get_logger(name)` | Returns a configured Python logger with timestamp + level + name + message format. |
| `cached_dataframe()` | Wraps `ingest.load_vendors_df()` with `@st.cache_resource` so the sheet is fetched only once per session. The sidebar's "Reload data from sheet" button clears this cache. |

## Data schema

The Google Sheet is the source of truth. Expected columns (case-sensitive):

| Column | Type | Used for |
|---|---|---|
| `Vendor_Name` | string | Display + fuzzy name lookup |
| `Vendor_Code` | string | Unique identifier — exact code lookup |
| `State` | string | Canonical filter (LLM-normalised on input) |
| `City` | string | Fuzzy filter |
| `Work_Type` | string | Lenient substring + fuzzy match |
| `Quality_Rating` | numeric (as string) | `min_rating` filter |
| `Recommendation` | string | Tag — derived from `overall_score` in the sheet |
| `overall_score` | numeric (as string) | Tier sort + tag derivation |
| `Travel_Radius_KM` | free text | Parsed by `parse_travel_km` |
| `Primary_Contact` | string | Display |
| `Email` | string | Display |
| `Bank Account Holder Name (Company)` | string | Bank section (Company) — gated |
| `Bank Account Number (Company)` | string | Bank section (Company) — gated |
| `IFSC Code (Company)` | string | Bank section (Company) — gated |
| `Bank Account Holder Name (Individual)` | string | Bank section (Individual) — gated |
| `Bank Account Number (Individual)` | string | Bank section (Individual) — gated |
| `IFSC Code (individual)` | string | Bank section (Individual) — gated |

### Score → tag mapping

| Score range | Tag | Sort priority |
|---|---|---|
| `score == 0` (or missing) | New / Never Used | 2 |
| `score ≥ 0.8` | Recommended | 0 |
| `0.5 ≤ score < 0.8` | Good | 1 |
| `0 < score < 0.5` | Risky | 3 |

The tag is set in the sheet itself; the executor doesn't recompute it. The score is what's actually used for sorting.

### Sensitive fields

The six bank-detail columns are treated as PII. They are stripped from the result before the LLM is called, **unless**:

1. `slots["requests_bank_details"] == True`, AND
2. `intent in ("lookup", "details")`, AND
3. Exactly one row matches the filters.

If any of those is false, the bank columns are dropped and `bank_clarification_needed` is set so the response generator asks for clarification.

## Key design decisions

### Why two LLM calls instead of one?

A single end-to-end LLM call ("here's the user's question and the entire DataFrame — answer it") would be slower, less accurate (token-limited), and would expose every field to the LLM including bank details. Splitting into slot-filler (input parser) and response-generator (output composer) keeps each call focused, lets us run deterministic logic in between, and provides a hard gate against bank-data leakage.

### Why pandas instead of vectors / RAG?

The dataset is small (~500 rows) and structured. Every meaningful question maps to a filter operation. Embeddings would add latency without adding value — fuzzy matching with `rapidfuzz` handles typos better and is interpretable (you can see exactly which canonical value matched and why).

### Why Streamlit-native cards instead of LLM-generated markdown?

Three reasons:
1. **No formatting drift** — the LLM cannot accidentally rearrange fields, miss a field, or invent one.
2. **Interactivity** — `st.expander` gives clean collapse/expand without custom CSS.
3. **Bank-detail safety** — the LLM never sees bank columns, so it can never accidentally include them in its prose. The bank section is rendered by Python code that runs after the executor's explicit authorisation.

### Why a separate `vendor_code` slot?

When the user types `VIJAYBILAVN`, naïvely fuzzy-matching against `Vendor_Name` finds nothing (it's not a name). Having a dedicated slot — combined with an executor fallback that retries any whitespace-free `vendor_name` against `Vendor_Code` — covers both the case where the LLM classifies correctly and the case where it doesn't.

### Why deterministic fallbacks?

Every LLM call (slot filler, response generator) has a fallback path. If the slot filler returns invalid JSON, defaults are used. If the response generator throws, `answer_formatter.format_result()` produces a plain-markdown answer from the same result dict. The user always gets *some* answer.

## Configuration

`.streamlit/secrets.toml` (and the same content in Streamlit Cloud's Secrets panel for deployment):

```toml
GROQ_API_KEY = "gsk_…"
GOOGLE_SHEET_ID = "…"
GOOGLE_SHEET_NAME = "VendorDB"

[gcp_service_account]
# Standard GCP service-account JSON, as TOML keys
type = "service_account"
project_id = "…"
private_key_id = "…"
private_key = "-----BEGIN PRIVATE KEY-----\n…\n-----END PRIVATE KEY-----\n"
client_email = "…@….iam.gserviceaccount.com"
…

[auth]
password = "…"
```

Tuning knobs (in `settings.py`):
- `FUZZ_THRESHOLD` — default 80. Lower = more permissive matching, more false positives. Higher = stricter, more false negatives.
- `LLM_MODEL` — default `"llama-3.3-70b-versatile"`. Any Groq-hosted model that supports JSON mode and the chat API will work.

## Testing

`pytest` suite in `tests/`. Covers:

- Fuzzy state/city matching including typos
- Recommendation tier filters and exclude-risky
- Work_Type matching with trailing words and multi-value cells
- Tier-aware sort ordering (Recommended → Good → New → Risky)
- Top-N limit and total_matches preservation
- `parse_travel_km` across the four real value shapes
- Bank-detail gating: stripped by default, included only for single-vendor lookups, clarification flagged for ambiguous queries
- Vendor code lookup: exact, case-insensitive, whitespace-padded
- Vendor name lookup: case-insensitive across the LLM's varied casing
- Self-correcting fallback: code-shaped value in the `vendor_name` slot
- Bank sections: Company-only, Individual-only, both present
- `<blank>` placeholder rendering for missing required bank fields
- Bank clarification message when vendor is ambiguous

Run with: `python -m pytest tests/ -v`
