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


def test_format_list_shows_vendor_code():
    df = pd.DataFrame([{
        "Vendor_Name": "Acme", "Vendor_Code": "ACME-001",
        "City": "Mumbai", "State": "MH",
        "Recommendation": "Recommended", "Primary_Contact": "1234",
    }])
    out = format_list(_result(df, {}))
    assert "ACME-001" in out


def test_format_lookup_groups_company_and_individual_bank():
    from src.answer_formatter import format_lookup
    df = pd.DataFrame([{
        "Vendor_Name": "Acme", "City": "Mumbai", "State": "MH",
        "Recommendation": "Good", "Primary_Contact": "1234",
        "Bank Account Holder Name (Company)":  "Acme Pvt Ltd",
        "Bank Account Number (Company)":       "1111111111",
        "IFSC Code (Company)":                 "HDFC0001",
        "Bank Account Holder Name (Individual)": "John Doe",
        "Bank Account Number (Individual)":      "2222222222",
        "IFSC Code (individual)":                "SBI0002",
    }])
    out = format_lookup(_result(df, {}))
    assert "Bank details (Company)" in out
    assert "Bank details (Individual)" in out
    assert "Acme Pvt Ltd" in out
    assert "John Doe" in out
    # Suffixes stripped from field labels
    assert "(Company)" not in out.split("Bank details (Company)")[1].split("Bank details (Individual)")[0]


def test_format_lookup_only_company_bank():
    from src.answer_formatter import format_lookup
    df = pd.DataFrame([{
        "Vendor_Name": "Acme", "City": "Mumbai", "State": "MH",
        "Primary_Contact": "1234",
        "Bank Account Holder Name (Company)": "Acme Pvt Ltd",
        "Bank Account Number (Company)":      "1111111111",
        "Bank Account Holder Name (Individual)": "",  # empty
        "Bank Account Number (Individual)":      "",
    }])
    out = format_lookup(_result(df, {}))
    assert "Bank details (Company)" in out
    assert "Bank details (Individual)" not in out


def test_format_lookup_bank_shows_blank_for_missing_required_field():
    from src.answer_formatter import format_lookup
    df = pd.DataFrame([{
        "Vendor_Name": "Acme", "City": "Mumbai", "State": "MH",
        "Primary_Contact": "1234",
        "Bank Account Holder Name (Company)": "Acme Pvt Ltd",
        "Bank Account Number (Company)":      "",  # missing
        "IFSC Code (Company)":                 "HDFC0001",
    }])
    out = format_lookup(_result(df, {}))
    assert "Bank details (Company)" in out
    # All 3 required labels are present even if one is blank
    assert "Bank Account Holder Name" in out
    assert "Bank Account Number" in out
    assert "IFSC Code" in out
    # The missing one is marked <blank>
    assert "<blank>" in out


def test_format_lookup_only_individual_bank():
    from src.answer_formatter import format_lookup
    df = pd.DataFrame([{
        "Vendor_Name": "Acme", "City": "Mumbai", "State": "MH",
        "Primary_Contact": "1234",
        "Bank Account Holder Name (Company)": "",
        "Bank Account Holder Name (Individual)": "John Doe",
        "Bank Account Number (Individual)":      "2222222222",
    }])
    out = format_lookup(_result(df, {}))
    assert "Bank details (Individual)" in out
    assert "Bank details (Company)" not in out
    assert "John Doe" in out


def test_format_lookup_shows_vendor_code():
    from src.answer_formatter import format_lookup
    df = pd.DataFrame([{
        "Vendor_Name": "Acme", "Vendor_Code": "ACME-001",
        "City": "Mumbai", "State": "MH", "Recommendation": "Good",
        "Primary_Contact": "1234",
    }])
    out = format_lookup(_result(df, {}))
    assert "ACME-001" in out


def test_format_list_new_never_used_badge():
    df = pd.DataFrame([{
        "Vendor_Name": "Fresh Co", "City": "X", "State": "Y",
        "Recommendation": "New / Never Used", "Primary_Contact": "1234",
    }])
    out = format_list(_result(df, {}))
    assert "New / Never Used" in out


def test_format_result_bank_clarification_path():
    from src.answer_formatter import format_result
    # Ambiguous bank request — multiple matches
    df = pd.DataFrame([
        {"Vendor_Name": "Arun M", "City": "Mumbai", "State": "MH", "Vendor_Code": "AM1"},
        {"Vendor_Name": "Arun S", "City": "Pune",   "State": "MH", "Vendor_Code": "AS2"},
    ])
    result = {
        "rows": df, "count": 2, "total_matches": 2,
        "applied_filters": {"Vendor_Name~": "Arun"},
        "intent": "lookup",
        "bank_clarification_needed": True,
        "bank_included": False,
    }
    out = format_result(result)
    assert "exact name" in out.lower() or "vendor code" in out.lower()
    assert "Arun M" in out
    assert "Arun S" in out


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
