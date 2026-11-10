"""Pulse 5 (Energy): Campus building behaviour & anomaly detection.
Dataset: I-BLEND (IIIT Delhi smart campus) - 52 months of multi-building power telemetry.
Signature challenge:
- Sensor dropouts and meter gaps (missing data).
- Transient spikes (electrical noise vs authentic institute events).
- 1-minute to hourly/daily resampling and alignment with institute academic calendar.
- Hostel late-night energy (00:00-05:00) during exams as a campus-wide proxy for sleep/stress.
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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

RAW = Path(__file__).resolve().parents[2] / "data" / "raw" / "iblend" / "energy_dataset"
PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed" / "energy"

DECISIONS = {
    "spike_handling":        ["clip_99th", "keep_raw", "rolling_median"],  # outlier filtering
    "meter_gap_imputation": ["linear_interpolate", "forward_fill", "drop"], # missing data handling
    "resample_freq":         ["1h", "2h", "daily"],                         # temporal resolution
}

FINDINGS = {
    "EN1": "Hostel late-night (00:00–05:00) energy consumption surges significantly during exam weeks (Watts difference)",
    "EN2": "Academic facility consumption on weekdays significantly exceeds weekend baseline (Watts surplus)",
    "EN3": "Daily load shape reconstruction error distinguishes anomalous days from regular academic schedules (residual gap)",
}


def load_raw() -> pd.DataFrame:
    parquet_path = PROCESSED / "hourly_campus_power.parquet"
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df
    csv_path = PROCESSED / "hourly_campus_power.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df
    raise FileNotFoundError(f"Processed energy data not found in {PROCESSED}")


def clean(decisions: dict, log: DecisionLog | None = None) -> pd.DataFrame:
    def _val(k, default):
        v = decisions.get(k, default)
        return v[0] if isinstance(v, list) else v

    log = log if log is not None else DecisionLog()
    df = load_raw().copy()
    n_init = len(df)
    log.step("load_hourly_telemetry", 0, n_init, "Loaded 52-month hourly campus building energy telemetry")

    power_cols = ["Academic", "Boys_main", "Boys_backup", "Girls_main", "Girls_backup", "total_hostel_power", "total_campus_power"]

    # 1. Spike / Outlier handling
    spike_mode = _val("spike_handling", "clip_99th")
    if spike_mode == "clip_99th":
        for col in power_cols:
            if col in df.columns:
                q99 = df[col].quantile(0.99)
                df[col] = df[col].clip(upper=q99)
        log.step("clip_99th_spikes", len(df), len(df), "Clipped transient electrical sensor spikes at 99th percentile", "clip_99th")
    elif spike_mode == "rolling_median":
        for col in power_cols:
            if col in df.columns:
                df[col] = df[col].rolling(3, min_periods=1, center=True).median()
        log.step("rolling_median_filter", len(df), len(df), "Applied 3-hour rolling median filter for spike suppression", "rolling_median")
    else:
        log.step("keep_raw_spikes", len(df), len(df), "Retained raw power readings without clipping sensor outliers", "keep_raw")

    # 2. Meter gap imputation
    impute_mode = _val("meter_gap_imputation", "linear_interpolate")
    before = len(df)
    if impute_mode == "drop":
        df = df.dropna(subset=["Academic", "total_hostel_power"]).copy()
        log.step("drop_meter_gaps", before, len(df), "Dropped hours with missing meter readings (complete cases only)", "drop")
    elif impute_mode == "forward_fill":
        df[power_cols] = df[power_cols].ffill().bfill()
        log.step("ffill_meter_gaps", len(df), len(df), "Forward-filled intermittent meter dropouts", "forward_fill")
    else:  # linear_interpolate
        df[power_cols] = df[power_cols].interpolate(method="linear").bfill().ffill()
        log.step("linear_interpolate_gaps", len(df), len(df), "Linearly interpolated meter communication dropouts", "linear_interpolate")

    # 3. Resampling frequency
    freq_mode = _val("resample_freq", "1h")
    if freq_mode != "1h":
        df = df.set_index("datetime")
        resampled_num = df[power_cols].resample("2h" if freq_mode == "2h" else "1D").mean()
        resampled_cat = df[["is_exam", "is_weekend", "is_late_night"]].resample("2h" if freq_mode == "2h" else "1D").max()
        df = pd.concat([resampled_num, resampled_cat], axis=1).reset_index()
        log.step(f"resample_{freq_mode}", before, len(df), f"Resampled temporal resolution to {freq_mode}", freq_mode)

    return df


def run_pipeline(decisions: dict) -> dict[str, float]:
    df = clean(decisions)

    # EN1: Hostel late-night energy during exams vs normal weeks
    if "is_late_night" in df.columns and (df["is_late_night"] == 1).any():
        late_night = df[df["is_late_night"] == 1]
        hostel_exam = late_night.loc[late_night["is_exam"] == 1, "total_hostel_power"].mean()
        hostel_norm = late_night.loc[late_night["is_exam"] == 0, "total_hostel_power"].mean()
    else:
        hostel_exam = df.loc[df["is_exam"] == 1, "total_hostel_power"].mean()
        hostel_norm = df.loc[df["is_exam"] == 0, "total_hostel_power"].mean()
    en1 = float(hostel_exam - hostel_norm)

    # EN2: Academic facility load: Weekday vs Weekend surplus
    acad_wkday = df.loc[df["is_weekend"] == 0, "Academic"].mean()
    acad_wkend = df.loc[df["is_weekend"] == 1, "Academic"].mean()
    en2 = float(acad_wkday - acad_wkend)

    # EN3: Anomaly reconstruction error separation
    if "datetime" in df.columns:
        df["date"] = pd.to_datetime(df["datetime"]).dt.date
        df["hour"] = pd.to_datetime(df["datetime"]).dt.hour
        piv = df.pivot_table(index="date", columns="hour", values="Academic").dropna()
        if len(piv) >= 30 and piv.shape[1] >= 12:
            X = piv.values
            Xs = StandardScaler().fit_transform(X)
            m_pca = PCA(n_components=min(3, Xs.shape[1]), random_state=0).fit(Xs)
            recon = m_pca.inverse_transform(m_pca.transform(Xs))
            errs = ((Xs - recon) ** 2).mean(axis=1)
            q90 = np.quantile(errs, 0.90)
            high_err = errs[errs >= q90]
            low_err = errs[errs < q90]
            en3 = float(high_err.mean() - low_err.mean()) if len(high_err) > 0 and len(low_err) > 0 else float("nan")
        else:
            en3 = float("nan")
    else:
        en3 = float("nan")

    return {"EN1": en1, "EN2": en2, "EN3": en3}


def run_stats(df: pd.DataFrame) -> dict:
    """Welch t-tests for exam period surge and weekday academic activity."""
    late_night = df[df["is_late_night"] == 1] if "is_late_night" in df.columns else df
    t_exam = stats_tests.welch_t(
        late_night.loc[late_night["is_exam"] == 1, "total_hostel_power"],
        late_night.loc[late_night["is_exam"] == 0, "total_hostel_power"],
    )
    t_weekday = stats_tests.welch_t(
        df.loc[df["is_weekend"] == 0, "Academic"],
        df.loc[df["is_weekend"] == 1, "Academic"],
    )
    return {"exam_t": t_exam, "weekday_t": t_weekday}


def get_daily_profile_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Shape transformation: Time series -> 24-hour daily load profile matrix (L11)."""
    df["date"] = pd.to_datetime(df["datetime"]).dt.date
    df["hour"] = pd.to_datetime(df["datetime"]).dt.hour
    piv = df.pivot_table(index="date", columns="hour", values="Academic").dropna()
    piv.columns = [f"hr_{h}" for h in piv.columns]
    return piv


def run_decomposition(df: pd.DataFrame, n_components: int = 3) -> dict:
    """Run PCA, TruncatedSVD, and t-SNE on daily load shapes (L12-14)."""
    X = get_daily_profile_matrix(df)
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


def run_clustering(df: pd.DataFrame, k: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cluster 24-hour daily load curves: regular weekday, weekend base, exam surge (L19-21)."""
    X = get_daily_profile_matrix(df)
    metrics, labels = clustering.run_all(X.values, k=k)
    km_labels = labels["kmeans"]
    sub_df = X.copy()
    sub_df["load_cluster"] = km_labels
    profiles = sub_df.groupby("load_cluster").mean().round(1)
    return metrics, profiles


def run_autoencoder(df: pd.DataFrame) -> np.ndarray:
    """Autoencoder anomaly detection on daily load profiles (L16)."""
    X = get_daily_profile_matrix(df)
    n_sample = min(300, len(X))
    return reduce.autoencoder_scores(X.iloc[:n_sample].values, bottleneck=4, epochs=30)


def mine_energy_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Apriori rules linking calendar events to high campus energy demand (L18)."""
    q_power = pd.qcut(df["total_campus_power"], 3, labels=["low_power", "med_power", "hi_power"])
    sub = pd.DataFrame({
        "exam": df["is_exam"].map({1: "exam_week", 0: "normal_week"}),
        "weekend": df["is_weekend"].map({1: "weekend", 0: "weekday"}),
        "power_demand": q_power,
    }).dropna()
    onehot = pd.get_dummies(sub)
    return rules.mine_rules(onehot, min_support=0.05, min_lift=1.1, min_conf=0.4, consequent_prefix="power_demand_hi_power")
