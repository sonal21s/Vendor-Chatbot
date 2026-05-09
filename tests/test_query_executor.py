import pandas as pd
from src.query_executor import execute, _fuzzy_match_value


def _sample_df():
    return pd.DataFrame([
        {"Vendor_Name": "Arun Maurya", "State": "Maharashtra", "City": "Mumbai",
         "Work_Type": "Plumbing", "Quality_Rating": "4", "Recommendation": "Recommended",
         "Primary_Contact": "9619853466"},
        {"Vendor_Name": "Jai Mata Enterprises", "State": "Maharashtra", "City": "Pune",
         "Work_Type": "Electrical", "Quality_Rating": "3", "Recommendation": "Good",
         "Primary_Contact": "8850827792"},
        {"Vendor_Name": "Akash Chaudhari", "State": "Karnataka", "City": "Bangalore",
         "Work_Type": "Plumbing", "Quality_Rating": "5", "Recommendation": "Risky",
         "Primary_Contact": "9422234930"},
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


def test_vendor_name_partial_match():
    result = execute(_sample_df(), {"intent": "lookup", "vendor_name": "Arun"})
    assert result["count"] == 1
    assert "Arun" in result["rows"].iloc[0]["Vendor_Name"]


def test_min_rating():
    result = execute(_sample_df(), {"intent": "list", "min_rating": 4})
    assert result["count"] == 2
