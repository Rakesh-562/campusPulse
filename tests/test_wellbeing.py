import pytest
from campuspulse.wellbeing import pipeline as P

def test_wellbeing_runs():
    try:
        P.load_raw()
    except FileNotFoundError:
        pytest.skip("Wellbeing data not present")
    r = P.run_pipeline({k: v[0] for k, v in P.DECISIONS.items()})
    assert set(r) == set(P.FINDINGS)
    assert r["W1"] > 0
    assert r["W2"] > 0
    assert r["W3"] > 0

