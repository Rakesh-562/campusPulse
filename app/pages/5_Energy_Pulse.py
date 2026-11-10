import sys, os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from campuspulse.energy import pipeline as P
from campuspulse.common.decisions import DecisionLog, robustness_matrix, heatmap
from campuspulse.common.plots import theme, badge, PULSE_COLOURS

theme()

st.set_page_config(page_title="Energy Pulse", page_icon="⚡", layout="wide")
st.title("⚡ ⑤ Energy Pulse: Campus Building Behaviour & Exam Period Surges")
st.caption("I-BLEND (IIIT Delhi, Indian smart campus) — 52 months of continuous 1-minute smart meter telemetry across academic and hostel facilities")

st.sidebar.header("🛠️ Cleaning Decisions (Person 5)")
decisions = {k: st.sidebar.selectbox(k, v) for k, v in P.DECISIONS.items()}

log = DecisionLog()
df_clean = P.clean(decisions, log)
res = P.run_pipeline(decisions)
stats_res = P.run_stats(df_clean)

# Metrics
m1, m2, m3 = st.columns(3)
m1.metric("EN1: Exam Late-Night Hostel Surge", f"+{res['EN1']:.1f} W", help="Late-night (00:00–05:00) power difference between exam periods and regular weeks")
m2.metric("EN2: Weekday Academic Surplus", f"+{res['EN2']:.1f} W", help="Surplus energy in academic building on working weekdays vs weekends")
m3.metric("EN3: Daily Load Anomaly Separation", f"{res['EN3']:.3f}" if pd.notna(res['EN3']) else "N/A", help="Autoencoder/PCA reconstruction error separation between anomalous and normal days")

st.markdown("---")
col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.subheader("Cross-Pulse Link: Hostel Late-Night Energy During Exams (L3–4)")
    st.write("Does the campus 'stay up' at exam time? Hostel late-night (00:00–05:00) power surges by over 2.1 kW during exams—an aggregate behavioral footprint of study stress and sleep deprivation.")
    
    late_night = df_clean[df_clean["is_late_night"] == 1] if "is_late_night" in df_clean.columns else df_clean
    fig, ax = plt.subplots(figsize=(6, 4))
    exam_labels = {0: "Normal Weeks", 1: "Exam Periods"}
    late_plot_df = late_night.copy()
    late_plot_df["Exam_Status"] = late_plot_df["is_exam"].map(exam_labels)
    sns.boxplot(data=late_plot_df, x="Exam_Status", y="total_hostel_power", ax=ax, palette=["#4a5568", "#805ad5"])
    badge(ax, "I-BLEND (IIIT Delhi)", real=True)
    ax.set_ylabel("Hostel Late-Night Power (Watts)")
    ax.set_title("Hostel Power (00:00–05:00): Normal vs Exam Weeks")
    st.pyplot(fig)

with col_chart2:
    st.subheader("Academic Facility Diurnal Load Profile (L5)")
    st.write("24-hour diurnal profile of academic buildings: sharp daytime ramp up on weekdays vs flat weekend baselines.")
    
    if "hour" in df_clean.columns:
        hourly_prof = df_clean.groupby(["hour", "is_weekend"])["Academic"].mean().reset_index()
        hourly_prof["Day_Type"] = hourly_prof["is_weekend"].map({0: "Weekday", 1: "Weekend"})
        
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.lineplot(data=hourly_prof, x="hour", y="Academic", hue="Day_Type", marker="o", ax=ax2, palette=["#2b6cb0", "#e53e3e"], lw=2)
        badge(ax2, "I-BLEND (IIIT Delhi)", real=True)
        ax2.set_xlabel("Hour of Day (0–23)")
        ax2.set_ylabel("Mean Academic Building Power (Watts)")
        ax2.set_title("Average 24-Hour Load Curve")
        ax2.set_xticks(range(0, 24, 2))
        st.pyplot(fig2)

st.markdown("---")
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Hypothesis Tests (Welch's t)",
    "📉 Load Shape PCA & t-SNE",
    "🤖 Autoencoder Anomaly Detection",
    "⚡ Daily Load Clustering Archetypes",
    "🔍 Association Rules Mining",
    "🔬 Finding Robustness Matrix"
])

with tab1:
    st.subheader("Hypothesis Testing on Building Behaviour")
    stats_data = [
        {"Hypothesis": "Exam Late-Night Hostel Surge (EN1)", "Test": "Welch's Two-Sample t-test", "Statistic": f"t = {stats_res['exam_t']['t']:.2f}", "p-value": f"{stats_res['exam_t']['p']:.2e}", "Effect Size": f"Cohen's d = {stats_res['exam_t']['cohens_d']:.3f}"},
        {"Hypothesis": "Weekday Academic Building Surplus (EN2)", "Test": "Welch's Two-Sample t-test", "Statistic": f"t = {stats_res['weekday_t']['t']:.2f}", "p-value": f"{stats_res['weekday_t']['p']:.2e}", "Effect Size": f"Cohen's d = {stats_res['weekday_t']['cohens_d']:.3f}"},
    ]
    st.dataframe(pd.DataFrame(stats_data), use_container_width=True)

with tab2:
    st.subheader("PCA and t-SNE of 24-Hour Load Profiles (L12–14)")
    decomp = P.run_decomposition(df_clean)
    c1, c2 = st.columns(2)
    with c1:
        st.write(f"**PCA Projection of Daily Load Curves (Exp. Var: {sum(decomp['pca_variance'][:2]):.1%}):**")
        fig_pca, ax_pca = plt.subplots(figsize=(6, 4))
        ax_pca.scatter(decomp["pca_pts"][:, 0], decomp["pca_pts"][:, 1], color="#805ad5", alpha=0.4, s=15)
        ax_pca.set_xlabel("PC1 (Base Load)"); ax_pca.set_ylabel("PC2 (Peak Amplitude)")
        badge(ax_pca, "I-BLEND", real=True)
        st.pyplot(fig_pca)
    with c2:
        st.write("**t-SNE 2D Manifold Embedding:**")
        fig_tsne, ax_tsne = plt.subplots(figsize=(6, 4))
        ax_tsne.scatter(decomp["tsne_pts"][:, 0], decomp["tsne_pts"][:, 1], color="#805ad5", alpha=0.7, s=20)
        badge(ax_tsne, "I-BLEND", real=True)
        st.pyplot(fig_tsne)

with tab3:
    st.subheader("Autoencoder Anomaly Detection on Daily Load Shapes (L16)")
    ae_scores = P.run_autoencoder(df_clean)
    fig_ae, ax_ae = plt.subplots(figsize=(7, 3.5))
    sns.histplot(ae_scores, bins=30, kde=True, ax=ax_ae, color="#805ad5")
    ax_ae.set_title("Autoencoder Reconstruction Error (Anomaly Score)")
    ax_ae.set_xlabel("Reconstruction MSE")
    badge(ax_ae, "I-BLEND", real=True)
    st.pyplot(fig_ae)

with tab4:
    st.subheader("Clustering 24-Hour Daily Load Curves (L19–21)")
    metrics, profiles = P.run_clustering(df_clean, k=3)
    st.write("**Clustering Algorithms Comparison (KMeans vs Agglomerative vs DBSCAN):**")
    st.dataframe(metrics, use_container_width=True)
    st.write("**Daily Load Curve Archetypes (Mean Hourly Power in Watts):**")
    st.dataframe(profiles, use_container_width=True)

with tab5:
    st.subheader("Apriori Association Rules (L18)")
    rules_df = P.mine_energy_rules(df_clean)
    st.dataframe(rules_df, use_container_width=True)

with tab6:
    st.subheader("Finding Robustness Matrix")
    v, s = robustness_matrix(P.run_pipeline, P.DECISIONS, P.FINDINGS)
    st.pyplot(heatmap(v, s, "Energy Pulse: Finding Robustness Heatmap"))
    st.caption(f"**Robustness Score:** {int(s.values.sum())} / {s.size} variants survive.")
