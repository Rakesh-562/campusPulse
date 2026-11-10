import pytest
from campuspulse.food import pipeline as P

def test_food_runs():
    try:
        P.load_raw()
    except FileNotFoundError:
        pytest.skip("NOVANIMAL data not present")
    r = P.run_pipeline({k: v[0] for k, v in P.DECISIONS.items()})
    assert set(r) == set(P.FINDINGS)
    assert r["F1"] > 0
    assert r["F2"] > 0
    assert r["F3"] > 0

