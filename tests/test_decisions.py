from campuspulse.common.decisions import robustness_matrix

def test_matrix_flags_fragile_finding():
    D = {"mode": ["a", "b"]}
    F = {"X": "stable", "Y": "fragile"}
    run = lambda c: {"X": 1.0, "Y": 1.0 if c["mode"] == "a" else -1.0}
    v, s = robustness_matrix(run, D, F)
    assert s.loc["X"].all() and not s.loc["Y", "mode=b"]
