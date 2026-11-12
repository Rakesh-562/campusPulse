import pytest
from campuspulse.academic import pipeline as P_acad
from campuspulse.wellbeing import pipeline as P_well
from campuspulse.energy import pipeline as P_ener

MODS = [P_acad, P_well, P_ener]


def test_all_pulses_defined():
    for mod in MODS:
        assert hasattr(mod, "DECISIONS") and hasattr(mod, "FINDINGS") and hasattr(mod, "run_pipeline")
        assert len(mod.DECISIONS) >= 2
        assert len(mod.FINDINGS) >= 3


def test_every_decision_changes_something():
    """A decision that never changes any finding is decoration, not analysis."""
    from campuspulse.common.decisions import robustness_matrix
    for mod in MODS:
        try:
            vals, _ = robustness_matrix(mod.run_pipeline, mod.DECISIONS, mod.FINDINGS)
        except FileNotFoundError:
            continue
        for dec, opts in mod.DECISIONS.items():
            cols = [f"{dec}={o}" for o in opts[1:]]
            changed = (vals[cols].sub(vals["DEFAULT"], axis=0).abs() > 1e-9).values.any()
            assert changed, f"{mod.__name__}: decision '{dec}' changes no finding"


def test_every_finding_has_evidence():
    """Each finding needs a graph (bars) and a simple hypothesis test for the dashboard."""
    for mod in MODS:
        try:
            ev = mod.finding_evidence()
        except FileNotFoundError:
            continue
        assert set(ev) == set(mod.FINDINGS)
        for fid, e in ev.items():
            assert e["bars"], f"{fid} has no bars"
            assert {"test", "null", "statistic", "p", "decision"} <= set(e["test"]), f"{fid} test incomplete"
