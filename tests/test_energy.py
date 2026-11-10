import pytest
from campuspulse.energy import pipeline as P

def test_energy_runs():
    try:
        P.load_raw()
    except FileNotFoundError:
        pytest.skip("I-BLEND data not present")
    r = P.run_pipeline({k: v[0] for k, v in P.DECISIONS.items()})
    assert set(r) == set(P.FINDINGS)
    assert r["EN1"] > 0
    assert r["EN2"] > 0
    assert r["EN3"] > 0

