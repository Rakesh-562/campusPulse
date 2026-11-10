"""Pulse 3 (Wellbeing): Student mental health and behavioral clustering.
Datasets:
- CDC YRBS (Youth Risk Behavior Survey): 10,710 respondents, behavioral questions.
- Malaysia University Mental Health Survey: 101 respondents, course text variations.
Signature challenge:
- Skipped sensitive questions (missing not at random).
- Dichotomous and ordinal coded survey responses.
- 49 messy spellings/abbreviations of degree courses (fuzzy text normalization).
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

PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed" / "wellbeing"
RAW_MH = Path(__file__).resolve().parents[2] / "data" / "raw" / "student_mh"

DECISIONS = {
    "missing_handling":     ["impute_mode", "drop", "indicator"],  # sensitive questions missingness
    "sleep_cutoff":         ["less_than_7hrs", "less_than_6hrs"], # severity threshold for sleep deprivation
    "course_standardization": ["fuzzy_canonical", "raw"],          # text normalization for Malaysia survey
}

FINDINGS = {
    "W1": "Sleep-deprived students report higher persistent sadness than well-rested peers (rate difference)",
    "W2": "Bullied students (school or electronic) report significantly higher sadness (rate difference)",
    "W3": "Physical inactivity co-occurs with elevated persistent sadness (rate difference)",
}


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    yrbs_file = PROCESSED / "yrbs_selected_questions.csv"
    if not yrbs_file.exists():
        yrbs_file = PROCESSED / "yrbs_xxhqn.csv"
    if not yrbs_file.exists():
        raise FileNotFoundError(f"YRBS processed data not found in {PROCESSED}")
    
    df_yrbs = pd.read_csv(yrbs_file)
    
    mh_file = RAW_MH / "Student Mental health.csv"
    df_mh = pd.read_csv(mh_file) if mh_file.exists() else pd.DataFrame()
    return df_yrbs, df_mh


def normalize_courses(series: pd.Series) -> pd.Series:
    """Fuzzy/canonical mapping for 49 disparate course strings in Malaysia survey (L9)."""
    def _map(val):
        if not isinstance(val, str):
            return "Other"
        s = val.strip().lower()
        if any(k in s for k in ["engin", "koe", "enm", "benl"]):
            return "Engineering"
        if any(k in s for k in ["bit", "bcs", "it", "cts", "mathem"]):
            return "Computing & IT"
        if any(k in s for k in ["law"]):
            return "Law"
        if any(k in s for k in ["islam", "irkhs", "kirkhs", "usuluddin", "taasl", "fiqh"]):
            return "Islamic Studies"
        if any(k in s for k in ["kenms", "account", "bank", "business", "econ"]):
            return "Business & Economics"
        if any(k in s for k in ["nurs", "biomed", "biotech", "radio", "marine"]):
            return "Health & Life Sciences"
        if any(k in s for k in ["psych", "human", "comm", "malcom", "mhsc", "ala", "kop", "tesl"]):
            return "Humanities & Social Sciences"
        return "Other"

    return series.apply(_map)


def clean(decisions: dict, log: DecisionLog | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    def _val(k, default):
        v = decisions.get(k, default)
        return v[0] if isinstance(v, list) else v

    log = log if log is not None else DecisionLog()
    df_yrbs, df_mh = load_raw()
    n_yrbs = len(df_yrbs)
    log.step("load_raw", 0, n_yrbs, "Loaded CDC YRBS survey and Malaysia MH dataset")

    # 1. Missing handling on YRBS
    missing_mode = _val("missing_handling", "impute_mode")
    key_cols = ["q26", "q88", "q23", "q24", "q79"]
    target_col = "q26"  # 1 = Sad, 2 = Not sad

    # Filter invalid target responses first
    valid_mask = df_yrbs[target_col].isin([1, 2])
    before = len(df_yrbs)
    df_yrbs = df_yrbs[valid_mask].copy()
    log.step("filter_valid_target", before, len(df_yrbs), "Retained respondents who answered sadness question (q26)", "valid_target")

    if missing_mode == "drop":
        before = len(df_yrbs)
        df_yrbs = df_yrbs.dropna(subset=key_cols).copy()
        log.step("drop_missing_sensitive", before, len(df_yrbs), "Complete case analysis: dropped rows with missing sensitive behaviors", "drop")
    elif missing_mode == "indicator":
        for col in key_cols:
            df_yrbs[f"{col}_missing"] = df_yrbs[col].isna().astype(int)
            mode_val = df_yrbs[col].mode().iloc[0]
            df_yrbs[col] = df_yrbs[col].fillna(mode_val)
        log.step("missing_indicator_mode", len(df_yrbs), len(df_yrbs), "Created missingness indicator columns and imputed mode", "indicator")
    else:  # impute_mode
        for col in key_cols:
            mode_val = df_yrbs[col].mode().iloc[0]
            df_yrbs[col] = df_yrbs[col].fillna(mode_val)
        log.step("impute_mode", len(df_yrbs), len(df_yrbs), "Imputed mode for missing responses on sensitive questions", "impute_mode")

    df_yrbs["sadness"] = (df_yrbs["q26"] == 1).astype(int)

    # 2. Sleep cutoff
    sleep_mode = _val("sleep_cutoff", "less_than_7hrs")
    if sleep_mode == "less_than_6hrs":
        df_yrbs["sleep_deprived"] = df_yrbs["q88"].isin([3, 4, 5]).astype(int)
    else:  # less_than_7hrs
        df_yrbs["sleep_deprived"] = df_yrbs["q88"].isin([2, 3, 4, 5]).astype(int)

    df_yrbs["bullied"] = ((df_yrbs["q23"] == 1) | (df_yrbs["q24"] == 1)).astype(int)
    df_yrbs["inactive"] = (df_yrbs["q79"] == 1).astype(int)

    # 3. Malaysia Course standardization
    if not df_mh.empty and "What is your course?" in df_mh.columns:
        course_mode = _val("course_standardization", "fuzzy_canonical")
        if course_mode == "fuzzy_canonical":
            df_mh["course_clean"] = normalize_courses(df_mh["What is your course?"])
            log.step("normalize_course_text", len(df_mh), len(df_mh), "Normalized 49 messy course variations into 7 academic disciplines", "fuzzy_canonical")
        else:
            df_mh["course_clean"] = df_mh["What is your course?"].astype(str).str.strip()
            log.step("keep_raw_course_text", len(df_mh), len(df_mh), "Retained raw course strings without fuzzy canonical grouping", "raw")

    return df_yrbs, df_mh


def run_pipeline(decisions: dict) -> dict[str, float]:
    df_yrbs, _ = clean(decisions)

    # W1: Sleep deprivation vs well-rested
    s_dep = df_yrbs.loc[df_yrbs["sleep_deprived"] == 1, "sadness"].mean()
    s_rest = df_yrbs.loc[df_yrbs["sleep_deprived"] == 0, "sadness"].mean()
    w1 = float(s_dep - s_rest)

    # W2: Bullied vs non-bullied
    b_yes = df_yrbs.loc[df_yrbs["bullied"] == 1, "sadness"].mean()
    b_no = df_yrbs.loc[df_yrbs["bullied"] == 0, "sadness"].mean()
    w2 = float(b_yes - b_no)

    # W3: Inactivity vs activity
    inact = df_yrbs.loc[df_yrbs["inactive"] == 1, "sadness"].mean()
    act = df_yrbs.loc[df_yrbs["inactive"] == 0, "sadness"].mean()
    w3 = float(inact - act)

    return {"W1": w1, "W2": w2, "W3": w3}


def run_stats(df_yrbs: pd.DataFrame) -> dict:
    """Hypothesis tests with Bonferroni correction: chi2, two proportion z, Cramér's V."""
    t_sleep = stats_tests.chi2_test(df_yrbs["sleep_deprived"], df_yrbs["sadness"])
    t_bully = stats_tests.chi2_test(df_yrbs["bullied"], df_yrbs["sadness"])
    t_inact = stats_tests.chi2_test(df_yrbs["inactive"], df_yrbs["sadness"])
    
    # Bonferroni multiple testing adjustment
    p_vals = [t_sleep["p"], t_bully["p"], t_inact["p"]]
    bonf = stats_tests.bonferroni(p_vals)

    return {"sleep_chi2": t_sleep, "bully_chi2": t_bully, "inactivity_chi2": t_inact, "bonferroni": bonf}


def get_behavior_matrix(df_yrbs: pd.DataFrame) -> pd.DataFrame:
    """Shape transformation: extract binary behavioral risk factors (L11)."""
    cols = ["sleep_deprived", "bullied", "inactive"]
    extra_cols = [c for c in ["q80", "q40", "q41"] if c in df_yrbs.columns]
    matrix = df_yrbs[cols + extra_cols].fillna(0)
    return pd.get_dummies(matrix)


def run_decomposition(df_yrbs: pd.DataFrame, n_components: int = 2) -> dict:
    """Run PCA, TruncatedSVD, and t-SNE on behavioral risk factors (L12-14)."""
    X = get_behavior_matrix(df_yrbs)
    pca_pts, pca_var, _ = reduce.pca(X, n=n_components)
    svd_pts, svd_var, _ = reduce.svd(X, n=n_components)
    n_sample = min(600, len(X))
    sample_idx = X.sample(n_sample, random_state=0).index
    tsne_pts = reduce.tsne(X.loc[sample_idx], perplexity=25)
    return {
        "pca_pts": pca_pts,
        "pca_variance": pca_var,
        "svd_pts": svd_pts,
        "svd_variance": svd_var,
        "tsne_pts": tsne_pts,
        "sample_idx": sample_idx,
    }


def run_clustering(df_yrbs: pd.DataFrame, k: int = 4) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cluster students into behavioral risk profiles (L19-21): healthy, isolated/bullied, exhausted, high-risk."""
    X = get_behavior_matrix(df_yrbs)
    n_sample = min(1200, len(X))
    sub_X = X.iloc[:n_sample]
    metrics, labels = clustering.run_all(sub_X.values, k=k)
    km_labels = labels["kmeans"]
    sub_df = df_yrbs.iloc[:n_sample].copy()
    sub_df["risk_cluster"] = km_labels
    profiles = sub_df.groupby("risk_cluster").agg(
        size=("sadness", "count"),
        sadness_rate=("sadness", "mean"),
        sleep_deprived_rate=("sleep_deprived", "mean"),
        bullied_rate=("bullied", "mean"),
        inactive_rate=("inactive", "mean"),
    ).round(3)
    return metrics, profiles


def mine_sadness_rules(df_yrbs: pd.DataFrame) -> pd.DataFrame:
    """Apriori association rules linking behavior combinations to persistent sadness (L18)."""
    cols = ["sleep_deprived", "bullied", "inactive", "sadness"]
    onehot = df_yrbs[cols].astype(bool)
    return rules.mine_rules(onehot, min_support=0.05, min_lift=1.1, min_conf=0.4, consequent_prefix="sadness")
