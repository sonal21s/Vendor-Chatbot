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
    assert names[-1] == "B"       # missing score last
