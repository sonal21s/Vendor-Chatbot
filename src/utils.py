import logging
import streamlit as st

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


@st.cache_resource(show_spinner="Loading vendor data…")
def cached_dataframe():
    from src.ingest import load_vendors_df
    return load_vendors_df()


@st.cache_resource(show_spinner="Loading city coordinates…")
def cached_city_coords():
    from src.ingest import load_city_coords
    return load_city_coords()
