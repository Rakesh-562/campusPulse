"""Pulse 4 (Food): Canteen food choice during vegetarian intervention.
Dataset: NOVANIMAL (ZHAW Zurich University of Applied Sciences, Switzerland).
26,340 transaction records across 2 university canteens over a 12-week field experiment (2017).
Signature challenge:
- Aggregated sales (point-of-sale till records rather than tracked individuals).
- Demand spikes and holidays (outliers vs authentic consumer demand).
- Day-of-week seasonality (smoothing).
- Randomised field experiment (alternating weeks): the one pulse that supports cause-and-effect claims.
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

RAW = Path(__file__).resolve().parents[2] / "data" / "raw" / "novanimal"
PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed" / "food"

DECISIONS = {
    # Lecture 5 noise removal. Mon-Fri attendance swings (+-80 meals/day) are noise for this question.
    "seasonality":     ["weekday_adjusted", "raw", "rolling_5day"],
    # Lecture 6-7 outliers at DAY level: 2 days fall outside the IQR fences.
    "outlier_filter":  ["keep", "iqr_drop_days", "winsorize_days"],
    "customer_filter": ["all_customers", "students_only", "staff_only"],   # patron subgroups
}

# Non-inferiority margin, fixed BEFORE looking at results: intervention weeks may sell
# at most 5% fewer meals per day than baseline weeks for us to call them "not worse".
NI_MARGIN_PCT = 0.05

FINDINGS = {
    "F1": "Vegetarian intervention weeks are non-inferior on daily sales (one-sided 95% lower bound of the "
          "difference + 5% margin; > 0 means proven not worse)",
    "F2": "Vegetarian/vegan share of meals rises in intervention weeks (share difference)",
    "F3": "Students choose vegetarian/vegan meals more often than staff (share difference)",
}

VEG_LABELS = ["Vegetarisch", "Pflanzlich", "Pflanzlich+"]


def load_raw() -> pd.DataFrame:
    parquet_path = PROCESSED / "canteen_sales.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    csv_raw = RAW / "2017_ZHAW_aggregated_menu_sales_NOVANIMAL.csv"
    if csv_raw.exists():
        df = pd.read_csv(csv_raw, sep=";")
        df["is_veg"] = df["label_content"].isin(VEG_LABELS).astype(int)
        df["date"] = pd.to_datetime(df["date"])
        return df
    raise FileNotFoundError(f"NOVANIMAL dataset not found in {PROCESSED} or {RAW}")


def _val(decisions, k, default):
    v = decisions.get(k, default)
    return v[0] if isinstance(v, list) else v


def _iqr_fences(s: pd.Series):
    q1, q3 = s.quantile([0.25, 0.75])
    return q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)


def clean(decisions: dict, log: DecisionLog | None = None) -> pd.DataFrame:
    """Transaction-level cleaning. Day-level smoothing/winsorizing happens in daily_series()."""
    log = log if log is not None else DecisionLog()
    df = load_raw().copy()
    log.step("load_raw", 0, len(df), "Loaded NOVANIMAL field experiment till sales (2 canteens, 60 days)")

    # 1. Customer subgroup filter
    cust_mode = _val(decisions, "customer_filter", "all_customers")
    before = len(df)
    if cust_mode == "students_only":
        df = df[df["rab_descript"] == "Studierende"].copy()
        log.step("filter_students", before, len(df), "Student transactions only", "students_only")
    elif cust_mode == "staff_only":
        df = df[df["rab_descript"] == "Mitarbeiter"].copy()
        log.step("filter_staff", before, len(df), "Staff transactions only", "staff_only")
    else:
        log.step("all_customers", before, len(df), "All customer groups kept", "all_customers")

    # 2. Day-level outliers (IQR on daily totals). Dropping removes whole days.
    out_mode = _val(decisions, "outlier_filter", "keep")
    daily_tot = df.groupby("date")["qty_weight"].sum()
    lo, hi = _iqr_fences(daily_tot)
    odd_days = daily_tot[(daily_tot < lo) | (daily_tot > hi)].index
    if out_mode == "iqr_drop_days":
        before = len(df)
        df = df[~df["date"].isin(odd_days)].copy()
        log.step("iqr_drop_days", before, len(df),
                 f"Dropped {len(odd_days)} day(s) outside IQR fences [{lo:.0f}, {hi:.0f}] meals/day", "iqr_drop_days")
    else:
        log.step("flag_outlier_days", len(df), len(df),
                 f"{len(odd_days)} day(s) outside IQR fences [{lo:.0f}, {hi:.0f}]: "
                 f"{'kept as real demand' if out_mode == 'keep' else 'winsorized at day level'}", out_mode)

    seas = _val(decisions, "seasonality", "weekday_adjusted")
    log.step("seasonality", len(df), len(df), {
        "weekday_adjusted": "Removed the weekday effect (each day minus its weekday mean); never mixes conditions",
        "raw": "Raw daily totals",
        "rolling_5day": "5-day rolling mean across dates: blends baseline and intervention days at week boundaries",
    }[seas], seas)
    return df


def daily_series(df: pd.DataFrame, decisions: dict) -> pd.DataFrame:
    """One row per date: total meals, vegetarian share, condition, after the chosen transforms."""
    d = df.assign(veg_qty=df["is_veg"] * df["qty_weight"]).groupby(["date", "condit"]).agg(
        total_sales=("qty_weight", "sum"), veg_sales=("veg_qty", "sum")).reset_index().sort_values("date")
    d["veg_share"] = d["veg_sales"] / d["total_sales"]

    if _val(decisions, "outlier_filter", "keep") == "winsorize_days":
        lo, hi = _iqr_fences(d["total_sales"])
        d["total_sales"] = d["total_sales"].clip(lo, hi)

    mode = _val(decisions, "seasonality", "weekday_adjusted")
    if mode == "weekday_adjusted":
        wd = d["date"].dt.dayofweek
        for c in ["total_sales", "veg_share"]:
            d[c] = d[c] - d.groupby(wd)[c].transform("mean") + d[c].mean()
    elif mode == "rolling_5day":
        for c in ["total_sales", "veg_share"]:
            d[c] = d[c].rolling(5, min_periods=1).mean()
    return d


def non_inferiority(inter: pd.Series, basis: pd.Series, margin_pct: float = NI_MARGIN_PCT) -> dict:
    """One-sided Welch non-inferiority test: H0 diff <= -margin vs H1 diff > -margin."""
    from scipy import stats as _st
    inter, basis = inter.dropna(), basis.dropna()
    diff = inter.mean() - basis.mean()
    v1, v2 = inter.var(ddof=1) / len(inter), basis.var(ddof=1) / len(basis)
    se = np.sqrt(v1 + v2)
    dof = (v1 + v2) ** 2 / (v1 ** 2 / (len(inter) - 1) + v2 ** 2 / (len(basis) - 1))
    margin = margin_pct * basis.mean()
    lower = diff - _st.t.ppf(0.95, dof) * se
    t = (diff + margin) / se
    return dict(diff=diff, se=se, lower_95=lower, margin=margin, t=t, p=_st.t.sf(t, dof),
                non_inferior=bool(lower > -margin))


def _veg_share(df: pd.DataFrame, group: str) -> float:
    g = df[df["rab_descript"] == group]
    return float((g["is_veg"] * g["qty_weight"]).sum() / g["qty_weight"].sum()) if len(g) else float("nan")


def run_pipeline(decisions: dict) -> dict[str, float]:
    df = clean(decisions)
    d = daily_series(df, decisions)
    inter, basis = d["condit"] == "Intervention", d["condit"] == "Basis"

    ni = non_inferiority(d.loc[inter, "total_sales"], d.loc[basis, "total_sales"])
    f1 = float(ni["lower_95"] + ni["margin"])                       # > 0  <=>  non-inferior
    f2 = float(d.loc[inter, "veg_share"].mean() - d.loc[basis, "veg_share"].mean())
    f3 = _veg_share(df, "Studierende") - _veg_share(df, "Mitarbeiter")  # NaN if a group was filtered out
    return {"F1": f1, "F2": f2, "F3": f3}


def run_stats(df: pd.DataFrame, decisions: dict | None = None) -> dict:
    """Randomised field experiment: non-inferiority on sales, Welch t on veg share, chi-square on groups."""
    decisions = decisions or {k: v[0] for k, v in DECISIONS.items()}
    d = daily_series(df, decisions)
    inter, basis = d["condit"] == "Intervention", d["condit"] == "Basis"
    t_sales = stats_tests.welch_t(d.loc[inter, "total_sales"], d.loc[basis, "total_sales"])
    t_share = stats_tests.welch_t(d.loc[inter, "veg_share"], d.loc[basis, "veg_share"])
    ni = non_inferiority(d.loc[inter, "total_sales"], d.loc[basis, "total_sales"])
    groups = df[df["rab_descript"].isin(["Studierende", "Mitarbeiter"])]
    chi_group = stats_tests.chi2_test(groups["rab_descript"], groups["is_veg"]) if len(groups) else {}
    return {"sales_t": t_sales, "veg_share_t": t_share, "sales_non_inferiority": ni, "group_veg_chi2": chi_group}


def extract_menu_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract keyword indicators from menu and meal descriptions (L9 Text Features)."""
    desc = df["article_description"].astype(str).str.lower()
    df_text = pd.DataFrame(index=df.index)
    df_text["is_buffet"] = desc.str.contains("buffet|hot and cold").astype(int)
    df_text["is_menu"] = desc.str.contains("menu").astype(int)
    return df_text


def get_daily_menu_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Shape transformation: pivot to daily sales per meal category (L11)."""
    piv = df.pivot_table(index="date", columns="label_content", values="qty_weight", aggfunc="sum").fillna(0)
    return piv


def run_decomposition(df: pd.DataFrame, n_components: int = 2) -> dict:
    """Run PCA, TruncatedSVD on daily meal choice distributions (L12-13)."""
    X = get_daily_menu_matrix(df)
    pca_pts, pca_var, _ = reduce.pca(X, n=n_components)
    svd_pts, svd_var, _ = reduce.svd(X, n=n_components)
    return {
        "pca_pts": pca_pts,
        "pca_variance": pca_var,
        "svd_pts": svd_pts,
        "svd_variance": svd_var,
    }


def run_clustering(df: pd.DataFrame, k: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cluster canteen days into demand profiles: heavy meat days, balanced days, veg surge days (L19-21)."""
    X = get_daily_menu_matrix(df)
    metrics, labels = clustering.run_all(X.values, k=k)
    km_labels = labels["kmeans"]
    sub_df = X.copy()
    sub_df["menu_cluster"] = km_labels
    profiles = sub_df.groupby("menu_cluster").mean().round(1)
    return metrics, profiles


def mine_food_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Apriori rules linking meal attributes and customer segments to vegetarian selection (L18)."""
    # One transaction per till record; the target is a single item "choice_veg" (vegetarian OR vegan),
    # so the three plant-based labels are not split into three weaker items.
    meal_records = pd.DataFrame({
        "week": df["condit"].map({"Basis": "baseline", "Intervention": "intervention"}),
        "customer": df["rab_descript"].map({"Studierende": "student", "Mitarbeiter": "staff"}).fillna("other"),
        "canteen": df["shop_description"].astype(str) if "shop_description" in df else "all",
        "day": df["date"].dt.day_name().str[:3],
        "choice": np.where(df["is_veg"] == 1, "veg", "not_veg"),
    }).dropna()
    onehot = pd.get_dummies(meal_records)
    return rules.mine_rules(onehot, min_support=0.03, min_lift=1.1, min_conf=0.4, consequent_prefix="choice_veg")
