import streamlit as st
from src.utils import cached_dataframe
from src.slot_filler import extract_slots
from src.query_executor import execute
from src.response_generator import generate_response

st.set_page_config(page_title="Vendor Assistant", page_icon="🏪", layout="centered")
st.title("Vendor Assistant")

df = cached_dataframe()

# Sidebar
with st.sidebar:
    st.header("Database")
    st.metric("Total vendors", len(df))
    if "State" in df.columns:
        st.metric("States covered", df["State"].nunique())
    if st.button("Reload data from sheet"):
        cached_dataframe.clear()
        st.rerun()
    st.divider()
    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.rerun()

# Init history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask about vendors…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        history_for_llm = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1]
        ]
        with st.spinner("Parsing query…"):
            slots = extract_slots(prompt, history=history_for_llm)
            result = execute(df, slots)
        with st.spinner("Composing answer…"):
            answer = generate_response(prompt, result, history=history_for_llm)

        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
