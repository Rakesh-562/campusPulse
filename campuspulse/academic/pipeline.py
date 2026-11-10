"""Pulse 1 (Academic): Early warning for university dropout.
Dataset: UCI 697 Predict Students' Dropout and Academic Success (4,424 rows, 37 columns).
Signature challenge:
- Grades of 0 that are really missing values (538 students; 92% dropped out).
- Outcome still unknown for 'Enrolled' students.
- Macroeconomic columns acting as cohort labels.
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

RAW = Path(__file__).resolve().parents[2] / "data" / "raw" / "uci_dropout"
GRADE = "Curricular units 1st sem (grade)"

DECISIONS = {
    "zero_grade": ["nan_plus_flag", "as_score", "drop"],   # hidden zeros are really missing
    "enrolled":   ["drop", "as_nondropout"],               # outcome unknown for 'Enrolled'
    "cohort_econ": ["keep", "drop"],                       # macroeconomic columns act as cohort tags
}

FINDINGS = {
    "A1": "Debtors drop out more than non-debtors (rate difference)",
    "A2": "Scholarship holders drop out less (rate difference)",
    "A3": "Dropouts have lower semester-1 grades (grade-point difference)",
}


def load_raw() -> pd.DataFrame:
    files = sorted(RAW.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV found in {RAW}")
    df = pd.read_csv(files[0], sep=None, engine="python")
    df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]
    return df


def clean(decisions: dict, log: DecisionLog | None = None) -> pd.DataFrame:
    def _val(k, default):
        v = decisions.get(k, default)
        return v[0] if isinstance(v, list) else v

    log = log if log is not None else DecisionLog()
    df = load_raw()
    n_init = len(df)
    log.step("load_raw", 0, n_init, "Initial raw UCI dropout cohort load")

    # 1. Enrolled status handling
    enrolled_mode = _val("enrolled", "drop")
    if enrolled_mode == "drop":
        before = len(df)
        df = df[df["Target"] != "Enrolled"].copy()
        log.step("remove_enrolled", before, len(df), "Censored students: graduation/dropout outcome not yet resolved", "drop")
    else:
        log.step("retain_enrolled", len(df), len(df), "Retain enrolled students as non-dropouts", "as_nondropout")

    df["dropout"] = (df["Target"] == "Dropout").astype(int)

    # 2. Hidden zero grade handling
    zero_mode = _val("zero_grade", "nan_plus_flag")
    zero = df[GRADE] == 0
    before = len(df)
    if zero_mode == "nan_plus_flag":
        df["grade_missing"] = zero.astype(int)
        df.loc[zero, GRADE] = np.nan
        log.step("zero_grade_imputation", before, len(df), f"{int(zero.sum())} zero grades treated as MNAR unsubmitted exams", "nan_plus_flag")
    elif zero_mode == "drop":
        df = df[~zero].copy()
        log.step("drop_zero_grades", before, len(df), "Dropped rows where semester 1 grade is 0", "drop")
    else:
        log.step("keep_zero_grades", before, len(df), "Retain 0.0 grades as literal evaluated scores", "as_score")

    # 3. Macroeconomic columns
    econ_cols = ["Unemployment rate", "Inflation rate", "GDP"]
    if _val("cohort_econ", "keep") == "drop":
        df = df.drop(columns=[c for c in econ_cols if c in df.columns])
        log.step("drop_macro_econ", len(df), len(df), "Remove cohort-level macroeconomic proxies", "drop")

    return df


def run_pipeline(decisions: dict) -> dict[str, float]:
    df = clean(decisions)
    r = lambda col, v: df.loc[df[col] == v, "dropout"].mean()

    # A1: Debtors drop out more than non-debtors
    a1 = float(r("Debtor", 1) - r("Debtor", 0))

    # A2: Scholarship holders drop out less
    a2 = float(r("Scholarship holder", 0) - r("Scholarship holder", 1))

    # A3: Dropouts have lower semester-1 grades
    grade_series = df[GRADE].dropna()
    valid_df = df.loc[grade_series.index]
    mean_persist = valid_df.loc[valid_df["dropout"] == 0, GRADE].mean()
    mean_dropout = valid_df.loc[valid_df["dropout"] == 1, GRADE].mean()
    a3 = float(mean_persist - mean_dropout)

    return {"A1": a1, "A2": a2, "A3": a3}


def run_stats(df: pd.DataFrame) -> dict:
    """Run hypothesis tests on academic findings: chi2, z, welch-t, ANOVA F."""
    t_debtor = stats_tests.chi2_test(df["Debtor"], df["dropout"])
    t_schol = stats_tests.chi2_test(df["Scholarship holder"], df["dropout"])
    valid_grade = df[[GRADE, "dropout"]].dropna()
    t_grade = stats_tests.welch_t(
        valid_grade.loc[valid_grade["dropout"] == 0, GRADE],
        valid_grade.loc[valid_grade["dropout"] == 1, GRADE],
    )
    # Two proportion z test
    n1 = int((df["Debtor"] == 1).sum())
    x1 = int(df.loc[df["Debtor"] == 1, "dropout"].sum())
    n2 = int((df["Debtor"] == 0).sum())
    x2 = int(df.loc[df["Debtor"] == 0, "dropout"].sum())
    z_debt = stats_tests.two_proportion_z(x1, n1, x2, n2)

    # ANOVA across top academic programmes
    top_courses = df["Course"].value_counts().index[:4]
    groups = [df.loc[df["Course"] == c, GRADE].dropna() for c in top_courses]
    anova_courses = stats_tests.anova_f(*groups)

    return {
        "debtor_chi2": t_debtor,
        "scholarship_chi2": t_schol,
        "grade_t": t_grade,
        "debtor_z": z_debt,
        "course_anova": anova_courses,
    }


def get_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Extract numeric features for dimensionality reduction and clustering."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    drop_cols = ["dropout", "Target"]
    features = [c for c in num_cols if c not in drop_cols]
    return df[features].fillna(df[features].median())


def detect_outliers(df: pd.DataFrame) -> dict:
    """Detect outliers in age and semester credits using IQR method (L6-7)."""
    age = df["Age at enrollment"] if "Age at enrollment" in df.columns else pd.Series()
    q25, q75 = age.quantile(0.25), age.quantile(0.75)
    iqr = q75 - q25
    outliers_age = df[age > q75 + 1.5 * iqr]
    return {"iqr_threshold": float(q75 + 1.5 * iqr), "outlier_count": len(outliers_age)}


def run_decomposition(df: pd.DataFrame, n_components: int = 2) -> dict:
    """Run PCA, TruncatedSVD, and t-SNE dimensionality reduction (L12-14)."""
    X = get_feature_matrix(df)
    pca_pts, pca_var, _ = reduce.pca(X, n=n_components)
    svd_pts, svd_var, _ = reduce.svd(X, n=n_components)
    # t-SNE on sample for speed
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


def run_clustering(df: pd.DataFrame, k: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run KMeans, Agglomerative, and DBSCAN to profile student academic personas (L19-21)."""
    X = get_feature_matrix(df)
    n_sample = min(1000, len(X))
    sub_X = X.iloc[:n_sample]
    metrics, labels = clustering.run_all(sub_X.values, k=k)
    km_labels = labels["kmeans"]
    sub_df = df.iloc[:n_sample].copy()
    sub_df["cluster"] = km_labels
    profiles = sub_df.groupby("cluster").agg(
        size=("dropout", "count"),
        dropout_rate=("dropout", "mean"),
        mean_grade=(GRADE, "mean"),
        debtor_rate=("Debtor", "mean"),
        scholarship_rate=("Scholarship holder", "mean"),
    ).round(3)
    return metrics, profiles


def run_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Mine Apriori association rules targeting student dropout (L18)."""
    cols = ["Debtor", "Scholarship holder", "dropout"]
    sub = df[[c for c in cols if c in df.columns]].dropna()
    onehot = pd.get_dummies(sub.astype(str))
    return rules.mine_rules(onehot, min_support=0.04, min_lift=1.1, min_conf=0.4, consequent_prefix="dropout_1")


def run_autoencoder(df: pd.DataFrame) -> np.ndarray:
    """Compute autoencoder anomaly reconstruction scores on student profiles (L16)."""
    X = get_feature_matrix(df)
    n_sample = min(300, len(X))
    return reduce.autoencoder_scores(X.iloc[:n_sample].values, bottleneck=4, epochs=30)
