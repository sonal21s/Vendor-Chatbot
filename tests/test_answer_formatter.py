import pandas as pd
from src.answer_formatter import format_count, format_list, format_lookup


def _result(rows, applied):
    return {"rows": rows, "count": len(rows), "applied_filters": applied}


def test_format_count():
    df = pd.DataFrame([{"Vendor_Name": "A"}, {"Vendor_Name": "B"}])
    out = format_count(_result(df, {"State": "Maharashtra"}))
    assert "2 vendors" in out
    assert "State=Maharashtra" in out


def test_format_list_empty():
    out = format_list(_result(pd.DataFrame(), {"State": "Mars"}))
    assert "No vendors match" in out


def test_format_list_renders_recommendation_tag_and_contact():
    df = pd.DataFrame([{
        "Vendor_Name": "Acme", "City": "Mumbai", "State": "MH",
        "Recommendation": "Recommended", "Quality_Rating": "5",
        "Primary_Contact": "9999999999",
    }])
    out = format_list(_result(df, {}))
    assert "Acme" in out
    assert "Recommended" in out
    assert "9999999999" in out
    assert "5" in out


def test_format_list_includes_work_type():
    df = pd.DataFrame([{
        "Vendor_Name": "Acme", "City": "Mumbai", "State": "MH",
        "Work_Type": "Plumbing", "Recommendation": "Good",
        "Primary_Contact": "1234",
    }])
    out = format_list(_result(df, {}))
    assert "Plumbing" in out


def test_format_list_risky_badge():
    df = pd.DataFrame([{
        "Vendor_Name": "RiskCo", "City": "X", "State": "Y",
        "Recommendation": "Risky", "Primary_Contact": "1234",
    }])
    out = format_list(_result(df, {}))
    assert "Risky" in out


def test_format_lookup_includes_contact():
    df = pd.DataFrame([{
        "Vendor_Name": "Acme", "City": "Mumbai", "State": "MH",
        "Recommendation": "Good", "Primary_Contact": "9876543210",
        "Email": "acme@example.com",
    }])
    out = format_lookup(_result(df, {"Vendor_Name~": "acme"}))
    assert "Acme" in out
    assert "Mumbai" in out
    assert "9876543210" in out
    assert "acme@example.com" in out
    assert "Good" in out
