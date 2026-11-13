import math
import pandas as pd
import pytest
from campuspulse.wellbeing import pipeline as P
from scripts.export_dashboard_data import weighted_distribution, weighted_rate


def test_wellbeing_runs():
    try:
        P.load_raw()
    except FileNotFoundError:
        pytest.skip("Wellbeing data not present")
    r = P.run_pipeline({k: v[0] for k, v in P.DECISIONS.items()})
    assert set(r) == set(P.FINDINGS)
    # Tests check the CODE, not the conclusions: a finding is allowed to come out negative.
    assert all(isinstance(v, float) and math.isfinite(v) for v in r.values())


def test_default_missing_handling_drops_missing_social_media_used_by_derived_indicator(monkeypatch):
    source = pd.DataFrame({
        "sad": [1, 2, 2],
        "sleep_hours": [6, 8, 8],
        "bullied_school": [2, 2, 2],
        "bullied_online": [2, 2, 2],
        "active_days": [0, 3, 3],
        "social_media_code": [6, None, 6],
        "sex": [1, 2, None],
        "weight": [1.0, 1.0, 1.0],
    })
    monkeypatch.setattr(P, "load_raw", lambda: source)

    cleaned = P.clean({key: options[0] for key, options in P.DECISIONS.items()})

    assert len(cleaned) == 1
    assert cleaned["heavy_social_media"].tolist() == [1]
    assert cleaned["female"].tolist() == [1]


def test_weighted_distribution_and_rate_report_weighted_shares_and_sample_counts():
    df = pd.DataFrame({"category": [0, 0, 1], "indicator": [0, 1, 1], "w": [1.0, 3.0, 6.0]})

    distribution = weighted_distribution(df, "category", {0: "No", 1: "Yes"})
    rate = weighted_rate(df, "indicator", "Positive")

    assert distribution == [
        {"label": "No", "share": 0.4, "n": 2},
        {"label": "Yes", "share": 0.6, "n": 1},
    ]
    assert rate == {"label": "Positive", "share": 0.9, "n": 3}
