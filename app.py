import streamlit as st
from src.utils import cached_dataframe, cached_city_coords
from src.slot_filler import extract_slots
from src.query_executor import execute
from src.response_generator import generate_response

st.set_page_config(page_title="Vendor Assistant", page_icon="🏪", layout="centered")

BANK_KEYWORDS = ("bank", "account", "ifsc", "upi", "branch")

# Required bank fields always shown when a bank section is rendered.
# Order is fixed; missing values render as <blank> so the user can see
# explicitly that the data isn't in the dataset rather than wondering
# whether we just chose not to display it.
REQUIRED_BANK_FIELDS = (
    "Bank Account Holder Name",
    "Bank Account Number",
    "IFSC Code",
)
BLANK = "_<blank>_"


def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        with st.form("login"):
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            if submitted:
                if password == st.secrets["auth"]["password"]:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Incorrect password")
        st.stop()


check_password()

st.title("Vendor Assistant")

df = cached_dataframe()
city_coords = cached_city_coords()


# ---------- card rendering --------------------------------------------------

def _clean_bank_label(col: str) -> str:
    return (
        str(col)
        .replace("(Company)", "")
        .replace("(company)", "")
        .replace("(Individual)", "")
        .replace("(individual)", "")
        .strip()
        .replace("_", " ")
    )


def _is_bank_col(col: str) -> bool:
    lower = str(col).lower()
    return any(k in lower for k in BANK_KEYWORDS)


def _classify_owner(col: str) -> str:
    lower = str(col).lower()
    if "(company)" in lower:
        return "company"
    if "(individual)" in lower:
        return "individual"
    return "other"


def _collect_bank_by_owner(row: dict) -> dict:
    """Return {owner: {clean_label: value, ...}} for all bank-keyword columns."""
    out = {"company": {}, "individual": {}, "other": {}}
    for col, val in row.items():
        if not _is_bank_col(col):
            continue
        owner = _classify_owner(col)
        label = _clean_bank_label(col)
        out[owner][label] = val if val not in (None, "") else ""
    return out


def _render_bank_section(title: str, fields_by_label: dict):
    """Render one bank section with the 3 required fields (showing <blank> if missing),
    plus any extra populated bank fields (e.g. Branch, UPI)."""
    st.markdown("")
    st.markdown(f"**{title}**")
    # Required fields, always in fixed order.
    for required in REQUIRED_BANK_FIELDS:
        val = fields_by_label.get(required, "")
        if not val:
            val = BLANK
        st.markdown(f"- {required}: {val}")
    # Any additional populated fields not in the required list.
    extras = {
        label: val for label, val in fields_by_label.items()
        if label not in REQUIRED_BANK_FIELDS and val
    }
    for label, val in extras.items():
        st.markdown(f"- {label}: {val}")


def render_vendor_card(row: dict, expanded: bool = False, key_suffix: str = ""):
    """Render a single vendor as a Streamlit expander."""
    name = row.get("Vendor_Name") or "Unknown"
    rec = row.get("Recommendation") or ""
    header = f"{name} — {rec}" if rec else name

    with st.expander(header, expanded=expanded):
        if row.get("Vendor_Code"):
            st.markdown(f"**Vendor Code:** {row['Vendor_Code']}")

        loc_parts = [p for p in [row.get("City"), row.get("State")] if p]
        if loc_parts:
            st.markdown(f"**Location:** {', '.join(loc_parts)}")

        if row.get("Work_Type"):
            st.markdown(f"**Work Type:** {row['Work_Type']}")
        if row.get("Quality_Rating"):
            st.markdown(f"**Quality Rating:** {row['Quality_Rating']}")
        if row.get("Travel_Radius_KM"):
            st.markdown(f"**Travel Radius:** {row['Travel_Radius_KM']} km")
        if row.get("Primary_Contact"):
            st.markdown(f"**Primary Contact:** {row['Primary_Contact']}")

        # Bank sections. The executor only includes bank columns when the
        # user explicitly asked AND the vendor is uniquely identified, so
        # if any bank column is present we render the full required format.
        bank_present = any(_is_bank_col(c) for c in row.keys())
        if bank_present:
            owners = _collect_bank_by_owner(row)
            section_titles = {
                "company": "Bank details (Company)",
                "individual": "Bank details (Individual)",
                "other": "Bank details",
            }
            rendered_any = False
            for owner in ("company", "individual", "other"):
                # Skip a section if NONE of its relevant fields have a value.
                # Within a rendered section, missing required fields render as <blank>.
                if owner == "other":
                    if not any(v for v in owners["other"].values()):
                        continue
                else:
                    if not any(owners[owner].get(label) for label in REQUIRED_BANK_FIELDS):
                        continue
                _render_bank_section(section_titles[owner], owners[owner])
                rendered_any = True
            if not rendered_any:
                st.markdown("")
                st.markdown("_No bank details on file for this vendor._")


def render_assistant_message(msg: dict):
    """Render an assistant message: intro text + (optionally) cards."""
    st.markdown(msg["intro"])
    if msg.get("rows"):
        expand_first = msg.get("expand_first", False)
        for i, row in enumerate(msg["rows"]):
            render_vendor_card(
                row,
                expanded=(i == 0 and expand_first),
                key_suffix=f"hist_{msg.get('turn', 0)}_{i}",
            )


# ---------- sidebar ---------------------------------------------------------

with st.sidebar:
    st.header("Database")
    st.metric("Total vendors", len(df))
    if "State" in df.columns:
        st.metric("States covered", df["State"].nunique())
    if st.button("Reload data from sheet"):
        cached_dataframe.clear()
        cached_city_coords.clear()
        st.rerun()
    st.divider()
    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.rerun()


# ---------- chat loop -------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and isinstance(msg.get("content"), dict):
            render_assistant_message(msg["content"])
        else:
            st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask about vendors…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        history_for_llm = []
        for m in st.session_state.messages[:-1]:
            content = m["content"]
            if isinstance(content, dict):
                content = content.get("intro", "")
            history_for_llm.append({"role": m["role"], "content": content})

        with st.spinner("Parsing query…"):
            slots = extract_slots(prompt, history=history_for_llm)
            result = execute(df, slots, city_coords)
        with st.spinner("Composing answer…"):
            intro = generate_response(prompt, result, history=history_for_llm)

        # Decide whether to show cards
        show_cards = (
            not result.get("bank_clarification_needed")
            and result.get("intent") != "count"
            and result["count"] > 0
        )
        rows_for_display = (
            result["rows"].to_dict(orient="records") if show_cards else []
        )
        expand_first = (
            result.get("intent") in ("lookup", "details") and result["count"] == 1
        )

        assistant_message = {
            "intro": intro,
            "rows": rows_for_display,
            "expand_first": expand_first,
            "turn": len(st.session_state.messages),
        }
        render_assistant_message(assistant_message)

    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_message}
    )
