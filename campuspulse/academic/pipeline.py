"""Pulse 1 (Academic): Early warning for university dropout.
Dataset: UCI 697 Predict Students' Dropout and Academic Success (4,424 rows, 37 columns).
Signature challenge:
- Grades of 0 that are really missing values (538 students enrolled in units but graded 0; 92% dropped out).
- 180 students with 0 enrolled units in semester 1 (credit transfers / special regimes): grade not applicable.
- Outcome still unknown for 'Enrolled' students (right-censored).
- Macroeconomic columns act as cohort labels (only 10 distinct combinations).
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
ENROLLED1 = "Curricular units 1st sem (enrolled)"
APPROVED1 = "Curricular units 1st sem (approved)"

# Integer columns that are CATEGORY CODES, not quantities (Course 9500 is not "bigger" than Course 33).
CODED = ["Marital status", "Application mode", "Course", "Previous qualification", "Nationality",
         "Mother's qualification", "Father's qualification", "Mother's occupation", "Father's occupation"]
ECON = ["Unemployment rate", "Inflation rate", "GDP"]

# What a university knows at each decision point (early-warning timeline).
STAGE_ENROL = ["Age at enrollment", "Admission grade", "Previous qualification (grade)", "Application order",
               "Gender", "Displaced", "Educational special needs", "International", "Scholarship holder",
               "Daytime/evening attendance"]
STAGE_SEM1 = ["Debtor", "Tuition fees up to date", "Curricular units 1st sem (credited)", ENROLLED1,
              "Curricular units 1st sem (evaluations)", APPROVED1, GRADE,
              "Curricular units 1st sem (without evaluations)"]
STAGE_SEM2 = [c.replace("1st", "2nd") for c in STAGE_SEM1 if "1st" in c]

DECISIONS = {
    "zero_grade": ["nan_plus_flag", "as_score", "drop"],   # hidden zeros are really missing
    "enrolled":   ["drop", "as_nondropout"],               # outcome unknown for 'Enrolled'
    "no_units_sem1": ["keep", "exclude"],                  # 180 students with 0 enrolled units (transfers?)
}

FINDINGS = {
    "A1": "Debtors drop out more than non-debtors (rate difference)",
    "A2": "Scholarship holders drop out less (rate difference)",
    "A3": "Dropouts have lower semester-1 grades (grade-point difference)",
    "A4": "Debt still predicts dropout WITHIN each semester-1 approval-rate band (mean rate difference)",
}


def load_raw() -> pd.DataFrame:
    files = sorted(RAW.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV found in {RAW}")
    df = pd.read_csv(files[0], sep=None, engine="python")
    df.columns = [c.strip().replace("﻿", "") for c in df.columns]
    return df.rename(columns={"Nacionality": "Nationality"})


def _val(decisions, k, default):
    v = decisions.get(k, default)
    return v[0] if isinstance(v, list) else v


def clean(decisions: dict, log: DecisionLog | None = None) -> pd.DataFrame:
    log = log if log is not None else DecisionLog()
    df = load_raw()
    log.step("load_raw", 0, len(df), "Initial raw UCI dropout cohort load")

    # 1. Enrolled status handling (right-censored outcome)
    if _val(decisions, "enrolled", "drop") == "drop":
        before = len(df)
        df = df[df["Target"] != "Enrolled"].copy()
        log.step("remove_enrolled", before, len(df), "Censored students: graduation/dropout outcome not yet resolved", "drop")
    else:
        log.step("retain_enrolled", len(df), len(df), "Retain enrolled students as non-dropouts", "as_nondropout")
    df["dropout"] = (df["Target"] == "Dropout").astype(int)

    # 2. Grades of 0: two different situations
    hidden = (df[GRADE] == 0) & (df[ENROLLED1] > 0)      # enrolled but never graded -> missing, not a score
    no_units = df[ENROLLED1] == 0                         # nothing to grade -> not applicable
    df["no_units_sem1"] = no_units.astype(int)
    df["approval_rate_sem1"] = np.where(df[ENROLLED1] > 0, df[APPROVED1] / df[ENROLLED1].replace(0, np.nan), np.nan)
    zero_mode = _val(decisions, "zero_grade", "nan_plus_flag")
    before = len(df)
    if zero_mode == "nan_plus_flag":
        df["grade_missing"] = hidden.astype(int)
        df.loc[hidden | no_units, GRADE] = np.nan
        log.step("zero_grade_to_missing", before, len(df),
                 f"{int(hidden.sum())} enrolled-but-ungraded zeros -> NaN + flag (MNAR); "
                 f"{int(no_units.sum())} with 0 enrolled units -> NaN (not applicable)", "nan_plus_flag")
    elif zero_mode == "drop":
        df = df[~(hidden | no_units)].copy()
        log.step("drop_zero_grades", before, len(df), "Dropped every row whose semester-1 grade is 0", "drop")
    else:
        df["grade_missing"] = 0
        log.step("keep_zero_grades", before, len(df), "Treat 0.0 as a real exam score", "as_score")

    # 3. Students with no enrolled semester-1 units: 75 of the 180 still graduated (credit transfers,
    #    special regimes?), so they are not simply early leavers. Keep them, or study only regular students.
    if _val(decisions, "no_units_sem1", "keep") == "exclude":
        before = len(df)
        df = df[df[ENROLLED1] > 0].copy()
        log.step("exclude_no_units", before, len(df), "Removed students with 0 enrolled semester-1 units", "exclude")
    else:
        log.step("keep_no_units", len(df), len(df), "Kept students with 0 enrolled semester-1 units (flagged)", "keep")

    # 4. Macroeconomic columns take only 10 distinct values = one per entry cohort. They are kept as a
    #    cohort_id for cohort-level tests and left OUT of every per-student feature matrix.
    df["cohort_id"] = df[ECON].astype(str).agg("|".join, axis=1).astype("category").cat.codes
    log.step("macro_as_cohort", len(df), len(df),
             f"Unemployment/inflation/GDP -> cohort_id ({df['cohort_id'].nunique()} cohorts); excluded from student features", "cohort_id")
    return df


def _rate(df, col, v):
    return df.loc[df[col] == v, "dropout"].mean()


def stratified_debt_effect(df: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """Debtor vs non-debtor dropout gap inside each semester-1 approval-rate band (finance vs grades)."""
    band = pd.cut(df["approval_rate_sem1"], [-0.01, 0.5, 0.99, 1.0], labels=["low (<=50%)", "mid", "all passed"])
    rows = []
    for b, g in df.groupby(band, observed=True):
        if g["Debtor"].nunique() == 2 and (g["Debtor"] == 1).sum() >= 10:
            rows.append(dict(band=b, n=len(g), n_debtors=int((g["Debtor"] == 1).sum()),
                             gap=_rate(g, "Debtor", 1) - _rate(g, "Debtor", 0),
                             cramers_v=stats_tests.chi2_test(g["Debtor"], g["dropout"])["cramers_v"]))
    tab = pd.DataFrame(rows)
    return (float(tab["gap"].mean()) if len(tab) else float("nan")), tab


def run_pipeline(decisions: dict) -> dict[str, float]:
    df = clean(decisions)
    a1 = float(_rate(df, "Debtor", 1) - _rate(df, "Debtor", 0))
    a2 = float(_rate(df, "Scholarship holder", 0) - _rate(df, "Scholarship holder", 1))
    g = df[[GRADE, "dropout"]].dropna()
    a3 = float(g.loc[g["dropout"] == 0, GRADE].mean() - g.loc[g["dropout"] == 1, GRADE].mean())
    a4, _ = stratified_debt_effect(df)
    return {"A1": a1, "A2": a2, "A3": a3, "A4": a4}


def run_stats(df: pd.DataFrame) -> dict:
    """Hypothesis tests: chi2, two-proportion z, Welch t, ANOVA F, stratified chi2 + Bonferroni."""
    t_debtor = stats_tests.chi2_test(df["Debtor"], df["dropout"])
    t_schol = stats_tests.chi2_test(df["Scholarship holder"], df["dropout"])
    valid_grade = df[[GRADE, "dropout"]].dropna()
    t_grade = stats_tests.welch_t(valid_grade.loc[valid_grade["dropout"] == 0, GRADE],
                                  valid_grade.loc[valid_grade["dropout"] == 1, GRADE])
    n1, x1 = int((df["Debtor"] == 1).sum()), int(df.loc[df["Debtor"] == 1, "dropout"].sum())
    n2, x2 = int((df["Debtor"] == 0).sum()), int(df.loc[df["Debtor"] == 0, "dropout"].sum())
    z_debt = stats_tests.two_proportion_z(x1, n1, x2, n2)
    top_courses = df["Course"].value_counts().index[:4]
    anova_courses = stats_tests.anova_f(*[df.loc[df["Course"] == c, GRADE].dropna() for c in top_courses])
    t_cohort = stats_tests.chi2_test(df["cohort_id"], df["dropout"])
    _, strat = stratified_debt_effect(df)
    pv = [t_debtor["p"], t_schol["p"], t_grade["p"], z_debt["p"], anova_courses["p"], t_cohort["p"]]
    return {"debtor_chi2": t_debtor, "scholarship_chi2": t_schol, "grade_t": t_grade, "debtor_z": z_debt,
            "course_anova": anova_courses, "cohort_chi2": t_cohort, "debt_within_grade_bands": strat,
            "bonferroni": stats_tests.bonferroni(pv)}


def detect_outliers(df: pd.DataFrame) -> dict:
    """IQR rule on age at enrolment (L6-7). Older students are real (mature students), so they are kept."""
    age = df["Age at enrollment"]
    q25, q75 = age.quantile([0.25, 0.75])
    hi = q75 + 1.5 * (q75 - q25)
    out = df[age > hi]
    return {"iqr_threshold": float(hi), "outlier_count": int(len(out)),
            "outlier_dropout_rate": float(out["dropout"].mean()) if len(out) else float("nan"),
            "others_dropout_rate": float(df.loc[age <= hi, "dropout"].mean())}


def get_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric QUANTITIES only (codes and IDs excluded), median-filled, for PCA / clustering."""
    num_cols = df.select_dtypes(include=[np.number]).columns
    features = [c for c in num_cols if c not in CODED + ECON + ["dropout", "cohort_id"]]
    X = df[features]
    return X.fillna(X.median())


def get_onehot_matrix(df: pd.DataFrame, min_share: float = 0.01) -> pd.DataFrame:
    """Coded categories -> one-hot (Lecture 12); categories under 1% grouped as 'other'."""
    cols = [c for c in CODED if c in df.columns]
    cat = df[cols].astype(str).apply(lambda s: s.where(s.map(s.value_counts(normalize=True)) >= min_share, "other"))
    return pd.get_dummies(cat).astype(float)


def run_decomposition(df: pd.DataFrame, n_components: int = 2) -> dict:
    """PCA on numeric quantities, TruncatedSVD on the one-hot coded categories, t-SNE (L12-14)."""
    X = get_feature_matrix(df)
    pca_pts, pca_var, _ = reduce.pca(X, n=n_components)
    svd_pts, svd_var, _ = reduce.svd(get_onehot_matrix(df).values, n=n_components)
    sample_idx = clustering.sample_rows(X, 500).index
    tsne_pts = reduce.tsne(X.loc[sample_idx], perplexity=30)
    return {"pca_pts": pca_pts, "pca_variance": pca_var, "svd_pts": svd_pts, "svd_variance": svd_var,
            "tsne_pts": tsne_pts, "sample_idx": sample_idx}


def run_clustering(df: pd.DataFrame, k: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """KMeans / Agglomerative / DBSCAN / OPTICS on a RANDOM 1,000-student sample (L19-21)."""
    X = clustering.sample_rows(get_feature_matrix(df), 1000)
    metrics, labels = clustering.run_all(X.values, k=k)
    sub_df = df.loc[X.index].copy()
    sub_df["cluster"] = labels["kmeans"]
    profiles = sub_df.groupby("cluster").agg(
        size=("dropout", "count"), dropout_rate=("dropout", "mean"), mean_grade=(GRADE, "mean"),
        debtor_rate=("Debtor", "mean"), scholarship_rate=("Scholarship holder", "mean"),
        mean_age=("Age at enrollment", "mean")).round(3)
    noise = labels["dbscan"] == -1
    if noise.any():
        profiles.loc["DBSCAN noise", ["size", "dropout_rate"]] = [int(noise.sum()), round(sub_df.loc[noise, "dropout"].mean(), 3)]
    return metrics, profiles


def run_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Apriori rules with dropout as the consequent (L18), on binned student attributes."""
    sub = pd.DataFrame({
        "debtor": df["Debtor"].map({1: "yes", 0: "no"}),
        "fees_up_to_date": df["Tuition fees up to date"].map({1: "yes", 0: "no"}),
        "scholarship": df["Scholarship holder"].map({1: "yes", 0: "no"}),
        "age": pd.cut(df["Age at enrollment"], [0, 20, 25, 100], labels=["<=20", "21-25", "26+"]),
        "sem1_approval": pd.cut(df["approval_rate_sem1"], [-0.01, 0.5, 0.99, 1.0], labels=["low", "mid", "all"]),
        "sem1_grade_missing": df.get("grade_missing", pd.Series(0, index=df.index)).map({1: "yes", 0: "no"}),
        "dropout": df["dropout"].map({1: "1", 0: "0"}),
    }).dropna()
    onehot = pd.get_dummies(sub)
    return rules.mine_rules(onehot, min_support=0.03, min_lift=1.2, min_conf=0.5, consequent_prefix="dropout_1")


def run_timeline(df: pd.DataFrame, top_share: float = 0.25, seed: int = 0) -> pd.DataFrame:
    """How early can we warn? Cross-validated logistic regression at three decision points.
    capture@25% = share of all dropouts found if the university contacts the 25% highest-risk students."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    y = df["dropout"].values
    onehot = get_onehot_matrix(df)
    stages = {"1 at enrolment": STAGE_ENROL, "2 after semester 1": STAGE_ENROL + STAGE_SEM1,
              "3 after semester 2": STAGE_ENROL + STAGE_SEM1 + STAGE_SEM2}
    rows = []
    for name, cols in stages.items():
        X = df[[c for c in cols if c in df.columns]].astype(float)
        X = pd.concat([X.fillna(X.median()), onehot], axis=1)
        if "grade_missing" in df:
            X["grade_missing"] = df["grade_missing"]
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        p = cross_val_predict(model, X.values, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                              method="predict_proba")[:, 1]
        top = p >= np.quantile(p, 1 - top_share)
        rows.append(dict(stage=name, n_features=X.shape[1], roc_auc=roc_auc_score(y, p),
                         capture_at_25pct=y[top].sum() / y.sum(), precision_at_25pct=y[top].mean(),
                         max_possible_capture=min(1.0, top.sum() / y.sum())))
    return pd.DataFrame(rows)


def run_autoencoder(df: pd.DataFrame) -> np.ndarray:
    """Autoencoder reconstruction error as an anomaly score on a random sample (L16)."""
    X = clustering.sample_rows(get_feature_matrix(df), 300)
    return reduce.autoencoder_scores(X.values, bottleneck=4, epochs=30)
