import streamlit as st

GROQ_API_KEY      = st.secrets["GROQ_API_KEY"]
GOOGLE_SHEET_ID   = st.secrets["GOOGLE_SHEET_ID"]
GOOGLE_SHEET_NAME = st.secrets.get("GOOGLE_SHEET_NAME", "VendorDB")
GCP_CREDENTIALS   = dict(st.secrets["gcp_service_account"])

LLM_MODEL  = "llama-3.3-70b-versatile"
FUZZ_THRESHOLD = 80  # rapidfuzz similarity threshold (0-100)
