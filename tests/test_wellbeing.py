import math
import pytest
from campuspulse.wellbeing import pipeline as P


def test_wellbeing_runs():
    try:
        P.load_raw()
    except FileNotFoundError:
        pytest.skip("Wellbeing data not present")
    r = P.run_pipeline({k: v[0] for k, v in P.DECISIONS.items()})
    assert set(r) == set(P.FINDINGS)
    # Tests check the CODE, not the conclusions: a finding is allowed to come out negative.
    assert all(isinstance(v, float) and math.isfinite(v) for v in r.values())
