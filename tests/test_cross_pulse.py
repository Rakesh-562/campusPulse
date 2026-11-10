import pytest
from campuspulse.academic import pipeline as P_acad
from campuspulse.engagement import pipeline as P_eng
from campuspulse.wellbeing import pipeline as P_well
from campuspulse.energy import pipeline as P_ener
from campuspulse.food import pipeline as P_food

def test_all_pulses_defined():
    for mod in [P_acad, P_eng, P_well, P_food, P_ener]:
        assert hasattr(mod, "DECISIONS")
        assert hasattr(mod, "FINDINGS")
        assert hasattr(mod, "run_pipeline")
        assert len(mod.DECISIONS) >= 2
        assert len(mod.FINDINGS) >= 3

def test_cross_pulse_findings_exist():
    # Academic & Engagement both capture dropout early warnings
    res_a = P_acad.run_pipeline({k: v[0] for k, v in P_acad.DECISIONS.items()})
    res_e = P_eng.run_pipeline({k: v[0] for k, v in P_eng.DECISIONS.items()})
    assert res_a["A1"] > 0 and res_e["E1"] > 0

    # Wellbeing & Energy both capture sleep / late-night stress
    res_w = P_well.run_pipeline({k: v[0] for k, v in P_well.DECISIONS.items()})
    res_en = P_ener.run_pipeline({k: v[0] for k, v in P_ener.DECISIONS.items()})
    assert res_w["W1"] > 0 and res_en["EN1"] > 0

