import pandas as pd

REC_LABELS = {
    "recommended": "Recommended",
    "good": "Good",
    "new / never used": "New / Never Used",
    "risky": "Risky",
}


def _rec_label(value: str) -> str:
    return REC_LABELS.get(str(value).strip().lower(), "")


def _contact_line(row: pd.Series) -> str:
    parts = []
    if row.get("Primary_Contact"):
        parts.append(f"Primary Contact: {row['Primary_Contact']}")
    if row.get("Email"):
        parts.append(f"Email: {row['Email']}")
    return " · ".join(parts)


def _filter_summary(applied: dict) -> str:
    if not applied:
        return "all vendors"
    parts = []
    for k, v in applied.items():
        if k == "Vendor_Name~":
            parts.append(f"name like \"{v}\"")
        elif k == "City_fallback":
            parts.append(
                f"no vendors in {v['requested']} — nearest: {v['nearest']} "
                f"(~{v['distance_km']} km)"
            )
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts)


def _is_bank_field(col_name: str) -> bool:
    lower = str(col_name).lower()
    return any(k in lower for k in ("bank", "account", "ifsc", "upi", "branch"))


def _classify_bank_field(col_name: str) -> str:
    lower = str(col_name).lower()
    if "(company)" in lower:
        return "company"
    if "(individual)" in lower:
        return "individual"
    return "other"


def _clean_bank_label(col_name: str) -> str:
    cleaned = (
        str(col_name)
        .replace("(Company)", "")
        .replace("(company)", "")
        .replace("(Individual)", "")
        .replace("(individual)", "")
        .strip()
    )
    return cleaned.replace("_", " ")


# Required bank fields shown in every rendered bank section. Missing values
# render as <blank> so users can see the data isn't on file rather than
# wondering whether we just hid it.
REQUIRED_BANK_FIELDS = (
    "Bank Account Holder Name",
    "Bank Account Number",
    "IFSC Code",
)
BLANK = "_<blank>_"


def format_count(result: dict) -> str:
    summary = _filter_summary(result["applied_filters"])
    return f"**{result['count']} vendors** matching: _{summary}_."


def _vendor_block(r: pd.Series) -> str:
    """Plain-text vendor block used by the fallback path."""
    name = r.get("Vendor_Name", "?")
    code = r.get("Vendor_Code", "")
    city = r.get("City", "")
    state = r.get("State", "")
    work_type = r.get("Work_Type", "")
    rating = r.get("Quality_Rating", "")
    rec = _rec_label(r.get("Recommendation", ""))
    contact = r.get("Primary_Contact", "")

    header = f"**{name}**"
    if rec:
        header += f" — {rec}"
    lines = [header]

    if code:
        lines.append(f"- Vendor Code: {code}")
    location = ", ".join(p for p in [city, state] if p)
    if location:
        lines.append(f"- Location: {location}")
    if work_type:
        lines.append(f"- Work Type: {work_type}")
    if rating:
        lines.append(f"- Quality Rating: {rating}")
    travel = r.get("Travel_Radius_KM", "")
    if travel:
        lines.append(f"- Travel Radius: {travel} km")
    if contact:
        lines.append(f"- Primary Contact: {contact}")

    return "\n".join(lines)


def format_list(result: dict, max_rows: int = 50) -> str:
    rows = result["rows"]
    summary = _filter_summary(result["applied_filters"])

    if rows.empty:
        return f"No vendors match: _{summary}_."

    header = f"**{len(rows)} vendors** matching: _{summary}_\n"
    if len(rows) > max_rows:
        header += f"\n_Showing first {max_rows} (of {len(rows)}):_\n"
        rows = rows.head(max_rows)

    blocks = [_vendor_block(r) for _, r in rows.iterrows()]
    return header + "\n\n".join(blocks)


def format_lookup(result: dict) -> str:
    rows = result["rows"]
    summary = _filter_summary(result["applied_filters"])

    if rows.empty:
        return f"No vendor found matching: _{summary}_."

    if len(rows) > 1:
        header = f"**{len(rows)} possible matches** for _{summary}_:\n\n"
        return header + format_list(result, max_rows=10).split("\n", 1)[1]

    r = rows.iloc[0]
    name = r.get("Vendor_Name", "Unknown")
    code = r.get("Vendor_Code", "")
    rec = _rec_label(r.get("Recommendation", ""))
    title = f"### {name}"
    if rec:
        title += f" — {rec}"
    lines = [title]
    if code:
        lines.append(f"Vendor Code: `{code}`")

    contact = _contact_line(r)
    if contact:
        lines.append(f"\n{contact}\n")

    skip = {"Vendor_Name", "Vendor_Code", "Primary_Contact", "Email", "Recommendation"}
    bank_by_owner = {"company": {}, "individual": {}, "other": {}}
    for col in r.index:
        if col in skip:
            continue
        val = r[col]
        if _is_bank_field(col):
            owner = _classify_bank_field(col)
            bank_by_owner[owner][_clean_bank_label(col)] = val if val not in (None, "") else ""
        elif val:
            lines.append(f"- **{col.replace('_', ' ')}:** {val}")

    section_titles = {
        "company": "**Bank details (Company)**",
        "individual": "**Bank details (Individual)**",
        "other": "**Bank details**",
    }
    bank_present_in_data = any(_is_bank_field(c) for c in r.index)
    if bank_present_in_data:
        rendered_any = False
        for owner in ("company", "individual", "other"):
            fields = bank_by_owner[owner]
            # Skip section entirely if NONE of the relevant fields have a value.
            if owner == "other":
                has_anything_populated = any(v for v in fields.values())
                if not has_anything_populated:
                    continue
            else:
                has_required_populated = any(
                    fields.get(label) for label in REQUIRED_BANK_FIELDS
                )
                if not has_required_populated:
                    continue
            lines.append("")
            lines.append(section_titles[owner])
            for required in REQUIRED_BANK_FIELDS:
                val = fields.get(required, "")
                lines.append(f"- {required}: {val if val else BLANK}")
            for label, val in fields.items():
                if label in REQUIRED_BANK_FIELDS or not val:
                    continue
                lines.append(f"- {label}: {val}")
            rendered_any = True
        if not rendered_any:
            lines.append("")
            lines.append("_No bank details on file for this vendor._")

    return "\n".join(lines)


def format_bank_clarification(result: dict) -> str:
    rows = result["rows"]
    if rows.empty:
        return (
            "I couldn't find that vendor. Please share the exact vendor name "
            "or Vendor Code so I can pull up their bank details."
        )
    if len(rows) > 1:
        names = []
        for _, r in rows.head(10).iterrows():
            n = r.get("Vendor_Name", "?")
            loc = ", ".join(p for p in [r.get("City", ""), r.get("State", "")] if p)
            code = r.get("Vendor_Code", "")
            tag = f" ({code})" if code else ""
            names.append(f"- **{n}**{tag} — {loc}")
        return (
            "I found multiple vendors that could match. Please confirm the exact name "
            "or Vendor Code — I'll only share bank details once you confirm:\n\n"
            + "\n".join(names)
        )
    return (
        "To share bank details, I need the exact vendor name or Vendor Code. "
        "Which vendor are you asking about?"
    )


def format_result(result: dict) -> str:
    if result.get("bank_clarification_needed"):
        return format_bank_clarification(result)
    intent = result.get("intent", "list")
    if intent == "count":
        return format_count(result)
    if intent in ("lookup", "details"):
        return format_lookup(result)
    return format_list(result)
