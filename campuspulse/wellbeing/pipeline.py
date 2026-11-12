"""Pulse 3 (Wellbeing): which behaviours co-occur with persistent sadness?

Datasets
- CDC YRBS 2023, national high-school survey (20,103 students). Built by scripts/preprocess_yrbs.py,
  which gives every question a plain column name (sad, sleep_hours, bullied_school, ...).

Main data challenges
- Skipped answers: about 1% skip the sadness question but 13% skip sleep and 24% skip social media.
- YRBS is a weighted sample, so unweighted rates are not national estimates.

Every finding is a signed number; positive means the claim holds.
"""
from __future__ import annotations
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
from pathlib import Path
import numpy as np
import pandas as pd
from campuspulse.common.decisions import DecisionLog
from campuspulse.common import stats_tests, reduce, clustering, rules

ROOT = Path(__file__).resolve().parents[2]
YRBS_FILE = ROOT / "data" / "processed" / "wellbeing" / "yrbs_2023.csv"

DECISIONS = {
    "missing_handling":       ["drop", "impute_mode", "indicator"],   # skipped behaviour questions
    "sleep_cutoff":           ["less_than_8hrs", "less_than_7hrs", "less_than_6hrs"],  # CDC uses < 8 h
    "survey_weights":         ["weighted", "unweighted"],
}

FINDINGS = {
    "W1": "Sleep-deprived students report persistent sadness more often (rate difference)",
    "W2": "Bullied students (at school or online) report persistent sadness more often (rate difference)",
    "W3": "Physically inactive students (0 active days a week) report persistent sadness more often (rate difference)",
}

SLEEP_LIMIT = {"less_than_8hrs": 8, "less_than_7hrs": 7, "less_than_6hrs": 6}
BEHAVIOUR_COLUMNS = ["sleep_hours", "bullied_school", "bullied_online", "active_days"]


def load_raw() -> pd.DataFrame:
    if not YRBS_FILE.exists():
        raise FileNotFoundError(f"{YRBS_FILE} not found. Run: python scripts/download_data.py --pulse yrbs "
                                f"then python scripts/preprocess_yrbs.py")
    return pd.read_csv(YRBS_FILE)


def clean(decisions: dict, log: DecisionLog | None = None) -> pd.DataFrame:
    log = log if log is not None else DecisionLog()
    decisions = {k: (v[0] if isinstance(v, list) else v) for k, v in decisions.items()}   # accept DECISIONS itself
    yrbs = load_raw()
    log.step("load_raw", 0, len(yrbs), "CDC YRBS 2023 national high-school survey")

    # 1. The outcome must be answered
    before = len(yrbs)
    yrbs = yrbs[yrbs["sad"].isin([1, 2])].copy()
    log.step("answered_sadness_question", before, len(yrbs), "Kept students who answered the sadness question", "required")

    # 2. Skipped behaviour questions
    mode = decisions.get("missing_handling", "drop")
    before = len(yrbs)
    if mode == "drop":
        yrbs = yrbs.dropna(subset=BEHAVIOUR_COLUMNS).copy()
        why = "Kept only students who answered every behaviour question (complete cases)"
    else:
        for col in BEHAVIOUR_COLUMNS:
            if mode == "indicator":
                yrbs[col + "_skipped"] = yrbs[col].isna().astype(int)
            yrbs[col] = yrbs[col].fillna(yrbs[col].mode().iloc[0])
        why = "Filled skipped answers with the most common answer" + (" and added a 'skipped' flag" if mode == "indicator" else "")
    log.step("skipped_answers", before, len(yrbs), why, mode)

    # 3. Readable 0/1 indicators
    limit = SLEEP_LIMIT[decisions.get("sleep_cutoff", "less_than_8hrs")]
    yrbs["sadness"] = (yrbs["sad"] == 1).astype(int)
    yrbs["sleep_deprived"] = (yrbs["sleep_hours"] < limit).astype(int)
    yrbs["bullied"] = ((yrbs["bullied_school"] == 1) | (yrbs["bullied_online"] == 1)).astype(int)
    yrbs["inactive"] = (yrbs["active_days"] == 0).astype(int)
    yrbs["heavy_social_media"] = (yrbs["social_media_code"] >= 6).astype(int)   # several times a day or more
    yrbs["female"] = (yrbs["sex"] == 1).astype(int)
    log.step("indicators", len(yrbs), len(yrbs), f"sleep_deprived = fewer than {limit} hours on a school night", "derived")

    # 4. Survey weights
    if decisions.get("survey_weights", "weighted") == "weighted":
        yrbs["w"] = yrbs["weight"].fillna(0)
        log.step("survey_weights", len(yrbs), len(yrbs), "Rates weighted with the CDC 'weight' column", "weighted")
    else:
        yrbs["w"] = 1.0
        log.step("survey_weights", len(yrbs), len(yrbs), "Unweighted sample rates", "unweighted")

    return yrbs


def sadness_rate(df: pd.DataFrame, mask) -> float:
    """Weighted share of students reporting persistent sadness within a group."""
    group = df[mask]
    return float(np.average(group["sadness"], weights=group["w"])) if group["w"].sum() > 0 else float("nan")


def run_pipeline(decisions: dict) -> dict[str, float]:
    d = clean(decisions)
    return {
        "W1": sadness_rate(d, d["sleep_deprived"] == 1) - sadness_rate(d, d["sleep_deprived"] == 0),
        "W2": sadness_rate(d, d["bullied"] == 1) - sadness_rate(d, d["bullied"] == 0),
        "W3": sadness_rate(d, d["inactive"] == 1) - sadness_rate(d, d["inactive"] == 0),
    }


def run_stats(d: pd.DataFrame) -> dict:
    """Chi-square tests (unweighted counts) with Bonferroni correction."""
    tests = {name: stats_tests.chi2_test(d[col], d["sadness"])
             for name, col in [("sleep_chi2", "sleep_deprived"), ("bully_chi2", "bullied"),
                               ("inactivity_chi2", "inactive"), ("social_media_chi2", "heavy_social_media")]}
    tests["bonferroni"] = stats_tests.bonferroni([t["p"] for t in tests.values()])
    return tests


def get_behavior_matrix(d: pd.DataFrame) -> pd.DataFrame:
    """One row per student, one 0/1 column per behaviour (L11 shape transformation)."""
    return d[["sleep_deprived", "bullied", "inactive", "heavy_social_media", "female"]].astype(float)


def run_decomposition(d: pd.DataFrame, n_components: int = 2) -> dict:
    X = get_behavior_matrix(d)
    pca_pts, pca_var, _ = reduce.pca(X, n=n_components)
    svd_pts, svd_var, _ = reduce.svd(X, n=n_components)
    sample_idx = clustering.sample_rows(X, 600).index
    return {"pca_pts": pca_pts, "pca_variance": pca_var, "svd_pts": svd_pts, "svd_variance": svd_var,
            "tsne_pts": reduce.tsne(X.loc[sample_idx], perplexity=25), "sample_idx": sample_idx}


def run_clustering(d: pd.DataFrame, k: int = 4) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = clustering.sample_rows(get_behavior_matrix(d), 1200)
    metrics, labels = clustering.run_all(X.values, k=k)
    sample = d.loc[X.index].assign(group=labels["kmeans"])
    profiles = sample.groupby("group").agg(
        size=("sadness", "count"), sadness_rate=("sadness", "mean"), sleep_deprived_rate=("sleep_deprived", "mean"),
        bullied_rate=("bullied", "mean"), inactive_rate=("inactive", "mean"),
        heavy_social_media_rate=("heavy_social_media", "mean"), female_share=("female", "mean")).round(3)
    return metrics, profiles


def mine_sadness_rules(d: pd.DataFrame) -> pd.DataFrame:
    """Apriori: which combinations of behaviours go with persistent sadness? (L18)"""
    items = d[["sleep_deprived", "bullied", "inactive", "heavy_social_media", "female", "sadness"]].astype(bool)
    return rules.mine_rules(items, min_support=0.04, min_lift=1.1, min_conf=0.4, consequent_prefix="sadness")


# ---------------------------------------------------------------- evidence for each finding (graph + simple test)

def _sadness_bars(d, column, labels):
    """Weighted share reporting persistent sadness per group. labels = {value: display name}."""
    return [dict(label=label, value=sadness_rate(d, d[column] == value), n=int((d[column] == value).sum()))
            for value, label in labels.items()]


def _proportion_test(d, column, name):
    """Two-proportion z-test on the sample counts (weights are used for the rates shown, not for the test)."""
    a, b = d[d[column] == 1], d[d[column] == 0]
    z = stats_tests.two_proportion_z(int(a["sadness"].sum()), len(a), int(b["sadness"].sum()), len(b))
    return stats_tests.test_result("Two-proportion z-test", f"Persistent sadness is equally common {name}", "z", z["z"], z["p"])


def finding_evidence(decisions: dict | None = None) -> dict:
    """For every finding: the bars its graph needs and one simple hypothesis test from the course."""
    d = clean(decisions or {k: v[0] for k, v in DECISIONS.items()})
    return {
        "W1": dict(kind="rate", unit="persistent sadness",
                   bars=_sadness_bars(d, "sleep_deprived", {1: "Sleep-deprived", 0: "Enough sleep"}),
                   test=_proportion_test(d, "sleep_deprived", "with and without enough sleep")),
        "W2": dict(kind="rate", unit="persistent sadness", bars=_sadness_bars(d, "bullied", {1: "Bullied", 0: "Not bullied"}),
                   test=_proportion_test(d, "bullied", "among bullied and not-bullied students")),
        "W3": dict(kind="rate", unit="persistent sadness", bars=_sadness_bars(d, "inactive", {1: "Inactive (0 days)", 0: "Active"}),
                   test=_proportion_test(d, "inactive", "among inactive and active students")),
    }


def surprising_rules(d: pd.DataFrame) -> list:
    """Associations people expect, which the data contradicts (lift < 1)."""
    items = pd.DataFrame({
        "heavy_social_media": d["heavy_social_media"] == 1, "enough_sleep": d["sleep_hours"] >= 8,
        "inactive": d["inactive"] == 1, "male": d["sex"] == 2, "sadness": d["sadness"] == 1,
    })
    return rules.surprising_rules(items, {
        "social_media": (["heavy_social_media", "enough_sleep"], "sadness",
                         "Heavy social-media users are sadder than other teenagers"),
        "inactive_boys": (["inactive", "male"], "sadness", "Teenagers who never exercise are sadder than average"),
    })
