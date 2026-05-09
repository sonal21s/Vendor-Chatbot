import pandas as pd

REC_BADGES = {
    "recommended": "✅ Recommended",
    "good": "🟡 Good",
    "risky": "⚠️ Risky",
}


def _rec_badge(value: str) -> str:
    return REC_BADGES.get(str(value).strip().lower(), "")


def _contact_line(row: pd.Series) -> str:
    parts = []
    if row.get("Primary_Contact"):
        parts.append(f"📞 {row['Primary_Contact']}")
    if row.get("Email"):
        parts.append(f"✉️ {row['Email']}")
    return " · ".join(parts)


def _filter_summary(applied: dict) -> str:
    if not applied:
        return "all vendors"
    parts = []
    for k, v in applied.items():
        if k == "Vendor_Name~":
            parts.append(f"name like \"{v}\"")
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts)


def format_count(result: dict) -> str:
    summary = _filter_summary(result["applied_filters"])
    return f"**{result['count']} vendors** matching: _{summary}_."


def format_list(result: dict, max_rows: int = 50) -> str:
    rows = result["rows"]
    summary = _filter_summary(result["applied_filters"])

    if rows.empty:
        return f"No vendors match: _{summary}_."

    header = f"**{len(rows)} vendors** matching: _{summary}_\n"
    if len(rows) > max_rows:
        header += f"\n_Showing first {max_rows} (of {len(rows)}):_\n"
        rows = rows.head(max_rows)

    lines = []
    for _, r in rows.iterrows():
        name = r.get("Vendor_Name", "?")
        city = r.get("City", "")
        state = r.get("State", "")
        work_type = r.get("Work_Type", "")
        rating = r.get("Quality_Rating", "")
        rec = _rec_badge(r.get("Recommendation", ""))
        contact = _contact_line(r)

        loc = " · ".join(p for p in [city, state] if p)
        meta_bits = []
        if work_type:
            meta_bits.append(f"🔧 {work_type}")
        if rating:
            meta_bits.append(f"⭐ {rating}")
        if rec:
            meta_bits.append(rec)

        line = f"- **{name}** — {loc}"
        if meta_bits:
            line += f" ({', '.join(meta_bits)})"
        if contact:
            line += f"  \n  {contact}"
        lines.append(line)

    return header + "\n".join(lines)


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
    rec = _rec_badge(r.get("Recommendation", ""))
    title = f"### {name}" + (f"  {rec}" if rec else "")
    lines = [title]

    # Show contact info prominently
    contact = _contact_line(r)
    if contact:
        lines.append(f"\n{contact}\n")

    # Show all remaining fields
    skip = {"Vendor_Name", "Primary_Contact", "Email", "Recommendation"}
    for col in r.index:
        if col in skip:
            continue
        val = r[col]
        if val:
            lines.append(f"- **{col.replace('_', ' ')}:** {val}")

    return "\n".join(lines)


def format_result(result: dict) -> str:
    intent = result.get("intent", "list")
    if intent == "count":
        return format_count(result)
    if intent in ("lookup", "details"):
        return format_lookup(result)
    return format_list(result)
