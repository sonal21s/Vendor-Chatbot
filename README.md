# Vendor Chatbot

A natural-language interface over a vendor database stored in Google Sheets. The service procurement team can ask questions in plain English ("how many recommended plumbers in Mumbai?", "tell me about Arun") and get instant, accurate answers with contact information.

Built with Streamlit, Groq (LLaMA 3.3 70B), pandas, and rapidfuzz.

## Architecture

```
User query
   │
   ▼
[ slot_filler ]   LLM extracts structured filters (state, city, work_type, recommendation tier, …)
   │              and normalizes state names ("andhra" → "Andhra Pradesh")
   ▼
[ query_executor ]  pandas + rapidfuzz applies filters with typo tolerance.
   │                Returns exact rows + count + applied filters.
   ▼
[ response_generator ]  LLM composes a tailored answer from the structured result,
   │                    formatting by intent (count / list / lookup).
   ▼
Markdown rendered in chat UI
```

The LLM is used only for **language understanding** (parsing the query) and **language generation** (phrasing the answer). All filtering and counting is deterministic pandas, so counts and lists are always exact — no top-k truncation.

## Local setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Copy the secrets template and fill in your credentials:
   ```
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   ```
   Edit `.streamlit/secrets.toml` with:
   - Your Groq API key from https://console.groq.com
   - Your Google Sheet ID and tab name
   - A Google Cloud service account with Sheets read access (share the sheet with the service account email)

3. Run:
   ```
   python -m streamlit run app.py
   ```

## Deploying to Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **New app** → select this repo → main file: `app.py`.
4. Under **Advanced settings → Secrets**, paste the entire contents of your local `.streamlit/secrets.toml` (the GCP service account block included).
5. Deploy. The app loads the sheet on first request and stays warm via Streamlit's resource cache.

## Project layout

```
app.py                      Streamlit UI + orchestration
settings.py                 Reads st.secrets, exposes typed constants
requirements.txt
src/
  ingest.py                 Google Sheets → pandas DataFrame
  slot_filler.py            LLM call: query → structured filters
  query_executor.py         pandas + rapidfuzz filtering
  response_generator.py     LLM call: result → conversational answer
  answer_formatter.py       Deterministic formatter (fallback if LLM fails)
  utils.py                  Logging + Streamlit caching helpers
tests/                      pytest suite for executor and formatter
```

## What the team can ask

- **Counts** — "how many vendors in Maharashtra?", "how many recommended plumbers?"
- **Lists** — "list good vendors in Pune", "show all electricians in Karnataka"
- **Lookups** — "tell me about Arun Maurya", "details for vendor code XYZ"
- **Combined filters** — "recommended plumbers in Mumbai with rating above 4"

Spelling is forgiving — "maharastra", "andhra", "MH", "tamil" all resolve to the canonical state. Vendor names support partial matches via fuzzy ratio.
