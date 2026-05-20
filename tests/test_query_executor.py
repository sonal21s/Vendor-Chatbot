import pandas as pd
from src.query_executor import execute, _fuzzy_match_value


def _sample_df():
    return pd.DataFrame([
        {"Vendor_Name": "Arun Maurya", "State": "Maharashtra", "City": "Mumbai",
         "Work_Type": "Plumbing", "Quality_Rating": "4", "Recommendation": "Recommended",
         "Primary_Contact": "9619853466", "overall_score": "0.85"},
        {"Vendor_Name": "Jai Mata Enterprises", "State": "Maharashtra", "City": "Pune",
         "Work_Type": "Electrical", "Quality_Rating": "3", "Recommendation": "Good",
         "Primary_Contact": "8850827792", "overall_score": "0.70"},
        {"Vendor_Name": "Akash Chaudhari", "State": "Karnataka", "City": "Bangalore",
         "Work_Type": "Plumbing", "Quality_Rating": "5", "Recommendation": "Risky",
         "Primary_Contact": "9422234930", "overall_score": "0.45"},
    ])


def test_fuzzy_match_handles_typo():
    candidates = ["Maharashtra", "Karnataka", "Gujarat"]
    assert _fuzzy_match_value("maharastra", candidates) == "Maharashtra"
    assert _fuzzy_match_value("karntaka", candidates) == "Karnataka"


def test_fuzzy_match_rejects_unrelated():
    candidates = ["Maharashtra", "Karnataka"]
    assert _fuzzy_match_value("xyz", candidates) is None


def test_count_by_state():
    result = execute(_sample_df(), {"intent": "count", "state": "Maharashtra"})
    assert result["count"] == 2
    assert result["applied_filters"]["State"] == "Maharashtra"


def test_count_with_typo():
    result = execute(_sample_df(), {"intent": "count", "state": "maharastra"})
    assert result["count"] == 2


def test_recommended_tag_filter():
    result = execute(_sample_df(), {"intent": "list", "recommendation": "Recommended"})
    assert result["count"] == 1
    assert result["rows"].iloc[0]["Vendor_Name"] == "Arun Maurya"


def test_good_tag_filter():
    result = execute(_sample_df(), {"intent": "list", "recommendation": "Good"})
    assert result["count"] == 1
    assert result["rows"].iloc[0]["Vendor_Name"] == "Jai Mata Enterprises"


def test_exclude_risky():
    result = execute(_sample_df(), {"intent": "list", "exclude_risky": True})
    assert result["count"] == 2
    assert "Akash Chaudhari" not in result["rows"]["Vendor_Name"].values


def test_combined_filters():
    result = execute(_sample_df(), {
        "intent": "list", "state": "Maharashtra", "recommendation": "Recommended",
    })
    assert result["count"] == 1
    assert result["rows"].iloc[0]["Vendor_Name"] == "Arun Maurya"


def test_work_type_strips_trailing_word():
    df = pd.DataFrame([
        {"Vendor_Name": "A", "State": "Andhra Pradesh", "Work_Type": "Interior Audit"},
        {"Vendor_Name": "B", "State": "Andhra Pradesh", "Work_Type": "Plumbing"},
    ])
    # User said "interior audit work" — trailing "work" must be stripped.
    result = execute(df, {"intent": "list", "work_type": "interior audit work"})
    assert result["count"] == 1
    assert result["rows"].iloc[0]["Vendor_Name"] == "A"


def test_work_type_matches_multi_value_cell():
    df = pd.DataFrame([
        {"Vendor_Name": "A", "Work_Type": "Interior Audit, Carpentry"},
        {"Vendor_Name": "B", "Work_Type": "Plumbing"},
    ])
    result = execute(df, {"intent": "list", "work_type": "interior audit"})
    assert result["count"] == 1
    assert result["rows"].iloc[0]["Vendor_Name"] == "A"


def test_vendor_name_partial_match():
    result = execute(_sample_df(), {"intent": "lookup", "vendor_name": "Arun"})
    assert result["count"] == 1
    assert "Arun" in result["rows"].iloc[0]["Vendor_Name"]


def test_min_rating():
    result = execute(_sample_df(), {"intent": "list", "min_rating": 4})
    assert result["count"] == 2


def test_parse_travel_km_handles_real_shapes():
    from src.query_executor import parse_travel_km
    import math
    assert parse_travel_km("Upto 250 km") == 250
    assert parse_travel_km("Local city within 50 km") == 50
    assert parse_travel_km("25 kilometres") == 25
    assert parse_travel_km("Anywhere in the country") == 9999
    assert parse_travel_km("Pan India") == 9999
    assert math.isnan(parse_travel_km(""))
    assert math.isnan(parse_travel_km("Local"))


def test_min_travel_km_filter_with_text_values():
    df = pd.DataFrame([
        {"Vendor_Name": "Near",      "City": "Mumbai", "Travel_Radius_KM": "Local city within 10 km"},
        {"Vendor_Name": "Mid",       "City": "Mumbai", "Travel_Radius_KM": "Upto 40 km"},
        {"Vendor_Name": "Far",       "City": "Mumbai", "Travel_Radius_KM": "100 kilometres"},
        {"Vendor_Name": "Unlimited", "City": "Mumbai", "Travel_Radius_KM": "Anywhere in the country"},
        {"Vendor_Name": "Empty",     "City": "Mumbai", "Travel_Radius_KM": ""},
    ])
    result = execute(df, {"intent": "list", "city": "Mumbai", "min_travel_km": 40})
    names = set(result["rows"]["Vendor_Name"])
    assert names == {"Mid", "Far", "Unlimited"}
    assert "Near" not in names      # parsed 10 < 40
    assert "Empty" not in names     # unparseable excluded


def test_min_travel_km_no_op_when_column_missing():
    df = pd.DataFrame([{"Vendor_Name": "A", "City": "Mumbai"}])
    result = execute(df, {"intent": "list", "min_travel_km": 40})
    # Column absent — filter should be a no-op, not crash or drop everything.
    assert result["count"] == 1


def test_results_sorted_best_to_worst_by_score():
    result = execute(_sample_df(), {"intent": "list"})
    names = list(result["rows"]["Vendor_Name"])
    # Highest score (0.85) first, lowest (0.45) last
    assert names == ["Arun Maurya", "Jai Mata Enterprises", "Akash Chaudhari"]


def test_limit_applies_after_sort():
    df = pd.DataFrame([
        {"Vendor_Name": f"V{i}", "overall_score": str(round(0.1 * i, 2))}
        for i in range(1, 11)  # scores 0.1 .. 1.0
    ])
    result = execute(df, {"intent": "list", "limit": 3})
    assert result["count"] == 3
    assert result["total_matches"] == 10
    # Top 3 should be the highest-scored
    assert list(result["rows"]["Vendor_Name"]) == ["V10", "V9", "V8"]


def test_no_limit_returns_all():
    df = pd.DataFrame([{"Vendor_Name": "A"}, {"Vendor_Name": "B"}])
    result = execute(df, {"intent": "list"})
    assert result["count"] == 2
    assert result["total_matches"] == 2


def test_sort_handles_missing_score():
    df = pd.DataFrame([
        {"Vendor_Name": "A", "overall_score": "0.5"},
        {"Vendor_Name": "B", "overall_score": ""},
        {"Vendor_Name": "C", "overall_score": "0.9"},
    ])
    result = execute(df, {"intent": "list"})
    names = list(result["rows"]["Vendor_Name"])
    assert names[0] == "C"        # highest score first
    # Empty score → tier "New / Never Used" (priority 2), Risky scores go last (priority 3).
    # Here B has empty score (New tier), A=0.5 is Good. Order should be: C (Rec), A (Good), B (New)
    assert names == ["C", "A", "B"]


def test_tier_order_recommended_good_new_risky():
    df = pd.DataFrame([
        {"Vendor_Name": "Rec",   "overall_score": "0.90"},
        {"Vendor_Name": "Risky", "overall_score": "0.30"},
        {"Vendor_Name": "Good",  "overall_score": "0.60"},
        {"Vendor_Name": "New",   "overall_score": "0"},
    ])
    result = execute(df, {"intent": "list"})
    assert list(result["rows"]["Vendor_Name"]) == ["Rec", "Good", "New", "Risky"]


def test_bank_columns_stripped_by_default():
    df = pd.DataFrame([
        {"Vendor_Name": "A", "Bank_Name": "HDFC", "Account_Number": "123", "IFSC": "HDFC0001"},
    ])
    result = execute(df, {"intent": "lookup", "vendor_name": "A"})
    assert "Bank_Name" not in result["rows"].columns
    assert "Account_Number" not in result["rows"].columns
    assert "IFSC" not in result["rows"].columns
    assert result["bank_included"] is False


def test_bank_included_for_single_vendor_lookup():
    df = pd.DataFrame([
        {"Vendor_Name": "Arun Maurya", "Bank_Name": "HDFC", "Account_Number": "123"},
    ])
    result = execute(df, {
        "intent": "lookup", "vendor_name": "Arun Maurya",
        "requests_bank_details": True,
    })
    assert "Bank_Name" in result["rows"].columns
    assert result["bank_included"] is True
    assert result["bank_clarification_needed"] is False


def test_bank_clarification_when_no_vendor_named():
    df = pd.DataFrame([
        {"Vendor_Name": "A", "Bank_Name": "HDFC"},
        {"Vendor_Name": "B", "Bank_Name": "SBI"},
    ])
    result = execute(df, {
        "intent": "list",  # not a lookup → can't expose bank
        "requests_bank_details": True,
    })
    assert result["bank_included"] is False
    assert result["bank_clarification_needed"] is True
    assert "Bank_Name" not in result["rows"].columns


def test_lookup_by_vendor_code():
    df = pd.DataFrame([
        {"Vendor_Name": "Arun Maurya",   "Vendor_Code": "ABCD-001"},
        {"Vendor_Name": "Other Person",  "Vendor_Code": "XYZ-999"},
    ])
    result = execute(df, {"intent": "lookup", "vendor_code": "ABCD-001"})
    assert result["count"] == 1
    assert result["rows"].iloc[0]["Vendor_Name"] == "Arun Maurya"


def test_lookup_by_vendor_code_case_insensitive():
    df = pd.DataFrame([
        {"Vendor_Name": "A", "Vendor_Code": "ArunJayprakaMumbaiVend"},
    ])
    result = execute(df, {"intent": "lookup", "vendor_code": "arunjayprakamumbaivend"})
    assert result["count"] == 1


def test_vendor_code_takes_precedence_over_name():
    df = pd.DataFrame([
        {"Vendor_Name": "Arun Maurya",  "Vendor_Code": "ABCD-001"},
        {"Vendor_Name": "Arun Sharma",  "Vendor_Code": "XYZ-999"},
    ])
    # Both provided — code wins.
    result = execute(df, {
        "intent": "lookup", "vendor_name": "Arun Sharma", "vendor_code": "ABCD-001"
    })
    assert result["count"] == 1
    assert result["rows"].iloc[0]["Vendor_Name"] == "Arun Maurya"


def test_bank_included_when_looked_up_by_code():
    df = pd.DataFrame([
        {"Vendor_Name": "A", "Vendor_Code": "C1",
         "Bank Account Number (Company)": "1234"},
    ])
    result = execute(df, {
        "intent": "lookup", "vendor_code": "C1",
        "requests_bank_details": True,
    })
    assert result["bank_included"] is True
    assert "Bank Account Number (Company)" in result["rows"].columns


def test_bank_clarification_when_lookup_returns_multiple():
    df = pd.DataFrame([
        {"Vendor_Name": "Arun Maurya",   "Bank_Name": "HDFC"},
        {"Vendor_Name": "Arun Sharma",   "Bank_Name": "SBI"},
    ])
    result = execute(df, {
        "intent": "lookup", "vendor_name": "Arun",
        "requests_bank_details": True,
    })
    assert result["bank_clarification_needed"] is True
    assert result["bank_included"] is False
    assert "Bank_Name" not in result["rows"].columns
