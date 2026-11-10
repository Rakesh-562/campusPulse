import pytest
from campuspulse.academic import pipeline as P

def test_academic_runs():
    try:
        P.load_raw()
    except FileNotFoundError:
        pytest.skip("UCI data not present")
    r = P.run_pipeline({k: v[0] for k, v in P.DECISIONS.items()})
    assert set(r) == set(P.FINDINGS)
