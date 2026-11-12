import numpy as np
import pandas as pd
from campuspulse.common.decisions import robustness_matrix, status_matrix, fragility_table, survival_summary


def test_matrix_flags_fragile_finding():
    D = {"mode": ["a", "b"]}
    F = {"X": "stable", "Y": "fragile"}
    run = lambda c: {"X": 1.0, "Y": 1.0 if c["mode"] == "a" else -1.0}
    v, s = robustness_matrix(run, D, F)
    assert s.loc["X"].all() and not s.loc["Y", "mode=b"]


def test_status_categories():
    v = pd.DataFrame({"DEFAULT": [1.0, 1.0, 1.0, 1.0, 1.0], "alt": [1.2, 0.3, 5.0, -1.0, np.nan]},
                     index=list("ABCDE"))
    st = status_matrix(v)["alt"].tolist()
    assert st == ["holds", "weakens", "inflates", "reverses", "n/a"]
    s = survival_summary(v)
    assert s["tested"] == 4 and s["holds"] == 1


def test_fragility_ranks_sign_flip_first():
    v = pd.DataFrame({"DEFAULT": [1.0, 1.0], "alt": [1.1, -0.5]}, index=["ok", "flip"])
    t = fragility_table(v)
    assert t.loc[0, "finding"] == "flip" and t.loc[0, "verdict"] == "fragile"
