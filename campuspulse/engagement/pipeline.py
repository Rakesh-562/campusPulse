"""Pulse 2 (Engagement): Online-learning disengagement.
Dataset: OULAD (Open University, UK) - 32,593 students, 22 course runs.
Signature challenge:
- 433 MB raw click logs aggregated to weekly student interactions.
- Noisy, spiky counts requiring smoothing.
- Missing assessment scores (students dropping out before assignment deadlines).
- Aligning different course presentation runs.
Positive finding value = the claim holds.
"""
from __future__ import annotations
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
from pathlib import Path
import numpy as np
import pandas as pd
from campuspulse.common.decisions import DecisionLog
from campuspulse.common import stats_tests, reduce, clustering, rules

RAW = Path(__file__).resolve().parents[2] / "data" / "raw" / "oulad"
PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed" / "engagement"

DECISIONS = {
    "click_smoothing": ["rolling_2wk", "raw", "exp_smoothing"],  # noise removal / smoothing
    "early_cutoff":    ["week_4", "week_2", "week_6"],          # early warning window
    "missing_scores":  ["impute_median", "impute_zero", "drop"],# missing assessment values
    "already_left":    ["exclude", "include"],                  # students who left before the cutoff (leakage)
}

FINDINGS = {
    "E1": "Disengaged learners in early weeks have higher withdrawal rates (bottom vs top click quartile)",
    "E2": "Students who persist achieve higher early assessment scores than those who withdraw",
    "E3": "Platform activity drop-off between week 1 and cutoff is steeper for withdrawn students",
}


def load_raw() -> pd.DataFrame:
    parquet_path = PROCESSED / "weekly_engagement.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    csv_path = PROCESSED / "weekly_engagement.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"Processed engagement dataset not found in {PROCESSED}")


def clean(decisions: dict, log: DecisionLog | None = None) -> pd.DataFrame:
    def _val(k, default):
        v = decisions.get(k, default)
        return v[0] if isinstance(v, list) else v

    log = log if log is not None else DecisionLog()
    df = load_raw().copy()
    n_init = len(df)
    log.step("load_processed", 0, n_init, "Loaded weekly aggregated OULAD student dataset")

    # 1. Missing assessment scores handling
    missing_mode = _val("missing_scores", "impute_median")
    before = len(df)
    if missing_mode == "drop":
        df = df.dropna(subset=["early_assessment_score"]).copy()
        log.step("drop_missing_scores", before, len(df), "Drop students missing early assessment grades", "drop")
    elif missing_mode == "impute_zero":
        df["early_assessment_score"] = df["early_assessment_score"].fillna(0.0)
        log.step("impute_zero_scores", before, len(df), "Impute missing scores with 0.0 (non-submission penalty)", "impute_zero")
    else:  # impute_median
        med = df["early_assessment_score"].median()
        df["early_assessment_score"] = df["early_assessment_score"].fillna(med)
        log.step("impute_median_scores", before, len(df), f"Impute missing scores with cohort median ({med:.1f})", "impute_median")

    # 2. Early cutoff week selection
    cutoff_str = _val("early_cutoff", "week_4")
    cutoff_week = int(str(cutoff_str).split("_")[1])

    # Leakage guard: a student who already unregistered before the cutoff has low clicks BECAUSE
    # they left. Predicting their withdrawal from those clicks is circular, so drop them.
    if "date_unregistration" in df.columns and _val("already_left", "exclude") == "exclude":
        before = len(df)
        left_early = pd.to_numeric(df["date_unregistration"], errors="coerce") < 7 * (cutoff_week + 1)
        df = df[~left_early].copy()
        log.step("drop_already_withdrawn", before, len(df),
                 f"Removed students who unregistered before the end of week {cutoff_week} (outcome leakage)", "exclude")
    elif "date_unregistration" in df.columns:
        log.step("keep_already_withdrawn", len(df), len(df),
                 "Kept students who had already left: their low clicks are a result of leaving, not a warning", "include")
    else:
        log.step("leakage_guard_unavailable", len(df), len(df),
                 "WARNING: no date_unregistration column in the processed file; students who left before the "
                 "cutoff are still included. Re-export it from studentRegistration.csv.", "leakage_guard")
    week_cols = [f"week_{w}" for w in range(cutoff_week + 1) if f"week_{w}" in df.columns]

    # 3. Smoothing of weekly clicks
    smoothing = _val("click_smoothing", "rolling_2wk")
    if smoothing == "rolling_2wk":
        for i in range(1, len(week_cols)):
            curr_col, prev_col = week_cols[i], week_cols[i - 1]
            df[f"smooth_{curr_col}"] = (df[curr_col] + df[prev_col]) / 2.0
        active_cols = [week_cols[0]] + [f"smooth_{col}" for col in week_cols[1:]]
        df["early_activity"] = df[active_cols].sum(axis=1)
        log.step("smoothing_rolling_2wk", len(df), len(df), "Applied 2-week rolling smoothing to eliminate weekly spikes", "rolling_2wk")
    elif smoothing == "exp_smoothing":
        vals = df[week_cols].values
        smoothed = np.zeros_like(vals, dtype=float)
        smoothed[:, 0] = vals[:, 0]
        for t in range(1, vals.shape[1]):
            smoothed[:, t] = 0.5 * vals[:, t] + 0.5 * smoothed[:, t - 1]
        df["early_activity"] = smoothed.sum(axis=1)
        log.step("smoothing_exponential", len(df), len(df), "Applied exponential moving average (alpha=0.5)", "exp_smoothing")
    else:
        df["early_activity"] = df[week_cols].sum(axis=1)
        log.step("smoothing_raw", len(df), len(df), "Retained raw click sums without temporal filtering", "raw")

    df["cutoff_week_col"] = f"week_{cutoff_week}"
    return df


def run_pipeline(decisions: dict) -> dict[str, float]:
    df = clean(decisions)
    
    # E1: Disengaged learners (bottom 25% early clicks) vs engaged (top 25%)
    q25 = df["early_activity"].quantile(0.25)
    q75 = df["early_activity"].quantile(0.75)
    r_low = df.loc[df["early_activity"] <= q25, "withdrawn"].mean()
    r_high = df.loc[df["early_activity"] >= q75, "withdrawn"].mean()
    e1 = float(r_low - r_high)

    # E2: Assessment score gap
    mean_pass = df.loc[df["withdrawn"] == 0, "early_assessment_score"].mean()
    mean_withdrawn = df.loc[df["withdrawn"] == 1, "early_assessment_score"].mean()
    e2 = float(mean_pass - mean_withdrawn)

    # E3: Activity drop-off between week 1 and cutoff week
    cutoff_col = df["cutoff_week_col"].iloc[0]
    w1_col = "week_1" if "week_1" in df.columns else "week_0"
    dropoff = (df[w1_col] - df[cutoff_col]) / (df[w1_col] + 10.0)
    e3 = float(dropoff[df["withdrawn"] == 1].mean() - dropoff[df["withdrawn"] == 0].mean())

    return {"E1": e1, "E2": e2, "E3": e3}


def run_stats(df: pd.DataFrame) -> dict:
    """Statistical hypothesis testing on engagement predictors: chi2, welch t, two proportion z."""
    q50 = df["early_activity"].median()
    df["low_clicks"] = (df["early_activity"] <= q50).astype(int)
    t_clicks = stats_tests.chi2_test(df["low_clicks"], df["withdrawn"])
    t_scores = stats_tests.welch_t(
        df.loc[df["withdrawn"] == 0, "early_assessment_score"],
        df.loc[df["withdrawn"] == 1, "early_assessment_score"],
    )
    # Two-proportion z test on low clicks vs high clicks
    n1 = int((df["low_clicks"] == 1).sum())
    x1 = int(df.loc[df["low_clicks"] == 1, "withdrawn"].sum())
    n2 = int((df["low_clicks"] == 0).sum())
    x2 = int(df.loc[df["low_clicks"] == 0, "withdrawn"].sum())
    z_res = stats_tests.two_proportion_z(x1, n1, x2, n2)

    return {"clicks_chi2": t_clicks, "scores_t": t_scores, "clicks_z": z_res}


def get_trajectory_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return wide weekly click matrix for trajectory clustering and PCA (L11)."""
    week_cols = [c for c in df.columns if c.startswith("week_") and not c.startswith("week_-") and not c.startswith("cutoff")]
    return df[week_cols].fillna(0.0)


def run_decomposition(df: pd.DataFrame, n_components: int = 2) -> dict:
    """Run PCA, TruncatedSVD, and t-SNE on weekly engagement trajectories (L12-14)."""
    X = get_trajectory_matrix(df)
    pca_pts, pca_var, _ = reduce.pca(X, n=n_components)
    svd_pts, svd_var, _ = reduce.svd(X, n=n_components)
    n_sample = min(500, len(X))
    sample_idx = X.sample(n_sample, random_state=0).index
    tsne_pts = reduce.tsne(X.loc[sample_idx], perplexity=30)
    return {
        "pca_pts": pca_pts,
        "pca_variance": pca_var,
        "svd_pts": svd_pts,
        "svd_variance": svd_var,
        "tsne_pts": tsne_pts,
        "sample_idx": sample_idx,
    }


def run_clustering(df: pd.DataFrame, k: int = 4) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cluster engagement trajectories into learner personas: steady, crammers, dropouts, lurkers (L19-21)."""
    # Random sample: the table is sorted by course module, so the first rows are almost all one course.
    sub_X = clustering.sample_rows(get_trajectory_matrix(df), 1200)
    metrics, labels = clustering.run_all(sub_X.values, k=k)
    km_labels = labels["kmeans"]
    sub_df = df.loc[sub_X.index].copy()
    sub_df["trajectory_cluster"] = km_labels
    profiles = sub_df.groupby("trajectory_cluster").agg(
        size=("withdrawn", "count"),
        withdrawal_rate=("withdrawn", "mean"),
        mean_early_clicks=("early_activity", "mean"),
        mean_score=("early_assessment_score", "mean"),
    ).round(2)
    return metrics, profiles


def run_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Apriori association rules linking early click and assessment quartiles to withdrawal (L18)."""
    q_clicks = pd.qcut(df["early_activity"], 3, labels=["low_clicks", "med_clicks", "hi_clicks"])
    q_score = pd.cut(df["early_assessment_score"], bins=[-1, 50, 75, 100], labels=["low_score", "med_score", "hi_score"])
    sub = pd.DataFrame({
        "clicks": q_clicks,
        "score": q_score,
        "withdrawn": df["withdrawn"].map({1: "withdrawn", 0: "persisted"}),
    }).dropna()
    onehot = pd.get_dummies(sub)
    return rules.mine_rules(onehot, min_support=0.05, min_lift=1.1, min_conf=0.4, consequent_prefix="withdrawn_withdrawn")


def run_timeline(cutoffs=("week_2", "week_4", "week_6"), top_share: float = 0.25) -> pd.DataFrame:
    """How early can we warn? Rank students by early clicks (fewest = highest risk) at each cutoff.
    capture@25% = share of all withdrawals found by contacting the 25% least active students.
    Same metric as campuspulse.academic.pipeline.run_timeline, so the two pulses can be plotted together."""
    from sklearn.metrics import roc_auc_score
    rows = []
    for cut in cutoffs:
        cfg = {**{k: v[0] for k, v in DECISIONS.items()}, "early_cutoff": cut}
        d = clean(cfg)
        y, risk = d["withdrawn"].values, -d["early_activity"].values
        top = risk >= np.quantile(risk, 1 - top_share)
        rows.append(dict(stage=cut, n_students=len(d), roc_auc=roc_auc_score(y, risk),
                         capture_at_25pct=y[top].sum() / y.sum(), precision_at_25pct=y[top].mean(),
                         max_possible_capture=min(1.0, top.sum() / y.sum())))
    return pd.DataFrame(rows)
