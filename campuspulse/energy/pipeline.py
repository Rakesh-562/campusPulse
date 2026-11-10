"""Pulse 5 (Energy): when does a campus building behave differently, and why?

Dataset: I-BLEND, IIIT Delhi (India). 52 months of electricity use for 9 meters, plus the institute calendar.
Built by scripts/preprocess_iblend.py into one row per hour.

What the calendar can and cannot tell us
- It marks each day as a teaching day (H) or a low-activity day (L: weekends, holidays, vacations).
- It does NOT mark exam days, so this pulse makes no claim about exams.

Main data challenges
- Meter gaps: the hostel meters sent no reading in about 25% of hours.
- Spikes: are very high readings glitches or real events?
- Season: Delhi's long vacation is also its hottest season (air-conditioning), which can fake a "term" effect.

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

DATA_FILE = Path(__file__).resolve().parents[2] / "data" / "processed" / "energy" / "hourly_campus_power.csv"
METERS = ["Academic", "Lecture", "Library", "Mess", "Facilities", "Boys_main", "Boys_backup", "Girls_main",
          "Girls_backup", "total_hostel_power", "total_campus_power"]

DECISIONS = {
    "meter_gap_imputation": ["drop", "linear_interpolate", "forward_fill"],   # hours with no reading
    "spike_handling":       ["keep_raw", "clip_99th", "rolling_median"],      # very high readings
    "resample_freq":        ["1h", "daily"],                                  # time resolution
}

FINDINGS = {
    "EN1": "The academic building draws more power on teaching days than on low-activity days (daytime 9:00-18:00, W)",
    "EN2": "Hostels draw more power at night (00:00-05:00) in term than in vacation (raw comparison, W)",
    "EN3": "The term-vs-vacation night difference survives comparing only within the same calendar month (season removed, W)",
}


def load_raw() -> pd.DataFrame:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"{DATA_FILE} not found. Run: python scripts/download_data.py --pulse iblend "
                                f"then python scripts/preprocess_iblend.py")
    df = pd.read_csv(DATA_FILE)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert("Asia/Kolkata")
    return df


def clean(decisions: dict, log: DecisionLog | None = None) -> pd.DataFrame:
    log = log if log is not None else DecisionLog()
    decisions = {k: (v[0] if isinstance(v, list) else v) for k, v in decisions.items()}   # accept DECISIONS itself
    df = load_raw()
    meters = [m for m in METERS if m in df.columns]
    log.step("load_hourly", 0, len(df), "Hourly mean power per meter, Aug 2013 - Dec 2017")

    # 1. Meter gaps
    gap_mode = decisions.get("meter_gap_imputation", "drop")
    before = len(df)
    if gap_mode == "drop":
        df = df.dropna(subset=["Academic", "total_hostel_power"]).copy()
        why = "Dropped hours where the academic or any hostel meter sent no reading"
    elif gap_mode == "linear_interpolate":
        df[meters] = df[meters].interpolate(limit_direction="both")
        why = "Filled gaps by drawing a straight line between the readings on either side"
    else:
        df[meters] = df[meters].ffill().bfill()
        why = "Filled gaps by repeating the last reading"
    log.step("meter_gaps", before, len(df), why, gap_mode)

    # 2. Spikes
    spike_mode = decisions.get("spike_handling", "keep_raw")
    if spike_mode == "clip_99th":
        for m in meters:
            df[m] = df[m].clip(upper=df[m].quantile(0.99))
        why = "Capped every meter at its 99th percentile"
    elif spike_mode == "rolling_median":
        for m in meters:
            df[m] = df[m].rolling(3, min_periods=1, center=True).median()
        why = "Replaced each hour by the median of itself and its two neighbours"
    else:
        why = "Kept every reading: no evidence that the high values are glitches"
    log.step("spikes", len(df), len(df), why, spike_mode)

    # 3. Time resolution
    if decisions.get("resample_freq", "1h") == "daily":
        flags = ["is_teaching", "is_vacation", "is_weekend"]
        df = (df.set_index("datetime")[meters + flags].resample("1D").mean().dropna(subset=["Academic"]).reset_index())
        df[flags] = df[flags].round().astype(int)
        log.step("resample_daily", before, len(df), "Averaged to one row per day (hour of day is lost)", "daily")
    return df


def _mean(df, mask, column):
    return float(df.loc[mask, column].mean())


def night_rows(df):
    """Hours 00:00-05:00, or every row when the data is daily (no hours left)."""
    return df[df["is_late_night"] == 1] if "is_late_night" in df else df


def term_vs_vacation_within_month(df: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """Term minus vacation night-time hostel power inside each month that has both; weighted by vacation hours."""
    nights = night_rows(df)
    month = nights["datetime"].dt.strftime("%Y-%m")
    rows = []
    for m, g in nights.groupby(month):
        term, vac = g[g["is_vacation"] == 0], g[g["is_vacation"] == 1]
        if len(term) and len(vac):
            rows.append(dict(month=m, term_rows=len(term), vacation_rows=len(vac),
                             term_mean=term["total_hostel_power"].mean(), vacation_mean=vac["total_hostel_power"].mean(),
                             diff=term["total_hostel_power"].mean() - vac["total_hostel_power"].mean()))
    table = pd.DataFrame(rows)
    if table.empty:
        return float("nan"), table
    return float(np.average(table["diff"], weights=table["vacation_rows"])), table


def run_pipeline(decisions: dict) -> dict[str, float]:
    df = clean(decisions)
    day = df[(df["hour"] >= 9) & (df["hour"] < 18)] if "hour" in df else df
    nights = night_rows(df)
    en1 = _mean(day, day["is_teaching"] == 1, "Academic") - _mean(day, day["is_teaching"] == 0, "Academic")
    en2 = _mean(nights, nights["is_vacation"] == 0, "total_hostel_power") - _mean(nights, nights["is_vacation"] == 1, "total_hostel_power")
    en3, _ = term_vs_vacation_within_month(df)
    return {"EN1": en1, "EN2": en2, "EN3": en3}


def run_stats(df: pd.DataFrame) -> dict:
    """Welch t-tests for the raw comparisons and a one-sample t-test on the same-month differences."""
    from scipy import stats as st
    day = df[(df["hour"] >= 9) & (df["hour"] < 18)] if "hour" in df else df
    nights = night_rows(df)
    teaching_t = stats_tests.welch_t(day.loc[day["is_teaching"] == 1, "Academic"], day.loc[day["is_teaching"] == 0, "Academic"])
    term_t = stats_tests.welch_t(nights.loc[nights["is_vacation"] == 0, "total_hostel_power"],
                                 nights.loc[nights["is_vacation"] == 1, "total_hostel_power"])
    _, months = term_vs_vacation_within_month(df)
    if len(months) >= 3:
        t, p = st.ttest_1samp(months["diff"], 0.0)
        within = dict(n_months=len(months), mean_diff=months["diff"].mean(), t=t, p=p,
                      share_months_positive=(months["diff"] > 0).mean())
    else:
        within = dict(n_months=len(months), mean_diff=np.nan, t=np.nan, p=np.nan, share_months_positive=np.nan)
    return {"teaching_t": teaching_t, "term_t": term_t, "within_month_t": within, "by_month": months,
            "bonferroni": stats_tests.bonferroni([x for x in [teaching_t["p"], term_t["p"], within["p"]] if pd.notna(x)])}


def get_daily_profile_matrix(df: pd.DataFrame, meter: str = "Academic") -> pd.DataFrame:
    """Shape transformation (L11): one row per day, 24 columns = average power in each hour."""
    d = df.assign(date=df["datetime"].dt.date, hour=df["datetime"].dt.hour)
    table = d.pivot_table(index="date", columns="hour", values=meter).dropna()
    table.columns = [f"hr_{h}" for h in table.columns]
    return table


def run_decomposition(df: pd.DataFrame, n_components: int = 3) -> dict:
    X = get_daily_profile_matrix(df)
    pca_pts, pca_var, _ = reduce.pca(X, n=n_components)
    svd_pts, svd_var, _ = reduce.svd(X, n=n_components)
    sample_idx = clustering.sample_rows(X, 500).index
    return {"pca_pts": pca_pts, "pca_variance": pca_var, "svd_pts": svd_pts, "svd_variance": svd_var,
            "tsne_pts": reduce.tsne(X.loc[sample_idx], perplexity=30), "sample_idx": sample_idx}


def run_clustering(df: pd.DataFrame, k: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Group days by the shape of their 24-hour academic-building load curve (L19-21)."""
    X = get_daily_profile_matrix(df)
    metrics, labels = clustering.run_all(X.values, k=k)
    calendar = df.assign(date=df["datetime"].dt.date).groupby("date")[["is_teaching", "is_vacation", "is_weekend"]].max()
    days = X.assign(group=labels["kmeans"]).join(calendar)
    profiles = days.groupby("group").agg(days=("hr_12", "size"), teaching_share=("is_teaching", "mean"),
                                         vacation_share=("is_vacation", "mean"), weekend_share=("is_weekend", "mean"),
                                         noon_power=("hr_12", "mean"), night_power=("hr_2", "mean")).round(3)
    return metrics, profiles


def run_autoencoder(df: pd.DataFrame) -> np.ndarray:
    """Autoencoder reconstruction error per day as an anomaly score (L16)."""
    X = clustering.sample_rows(get_daily_profile_matrix(df), 300)
    return reduce.autoencoder_scores(X.values, bottleneck=4, epochs=30)


def delhi_season(month: int) -> str:
    return {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "summer", 5: "summer", 6: "summer",
            7: "monsoon", 8: "monsoon", 9: "monsoon", 10: "autumn", 11: "autumn"}[month]


def mine_energy_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Apriori (L18): which calendar, time-of-day and season combinations go with high campus demand?"""
    items = pd.DataFrame({
        "day": df["is_teaching"].map({1: "teaching_day", 0: "low_activity_day"}),
        "period": df["is_vacation"].map({1: "vacation", 0: "term"}),
        "time": pd.cut(df["datetime"].dt.hour, [-1, 5, 9, 17, 23], labels=["night", "morning", "daytime", "evening"]),
        "season": df["datetime"].dt.month.map(delhi_season),
        "power_demand": pd.qcut(df["total_campus_power"], 3, labels=["low_power", "med_power", "hi_power"]),
    }).dropna()
    return rules.mine_rules(pd.get_dummies(items), min_support=0.03, min_lift=1.2, min_conf=0.5,
                            consequent_prefix="power_demand_hi_power")
