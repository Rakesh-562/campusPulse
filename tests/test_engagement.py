import pytest
from campuspulse.engagement import pipeline as P

def test_engagement_runs():
    try:
        P.load_raw()
    except FileNotFoundError:
        pytest.skip("OULAD data not present")
    r = P.run_pipeline({k: v[0] for k, v in P.DECISIONS.items()})
    assert set(r) == set(P.FINDINGS)
    assert r["E1"] > 0
    assert r["E2"] > 0
    assert r["E3"] > 0

