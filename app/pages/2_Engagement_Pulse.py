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

from campuspulse.engagement import pipeline as P
from campuspulse.common.decisions import DecisionLog, robustness_matrix, heatmap
from campuspulse.common.plots import theme, badge, PULSE_COLOURS

theme()

st.set_page_config(page_title="Engagement Pulse", page_icon="💻", layout="wide")
st.title("💻 ② Engagement Pulse: Online Learner Disengagement")
st.caption("OULAD (Open University, UK) — 32,593 students, 22 course presentations, 10.6M clicks aggregated to weekly metrics")

st.sidebar.header("🛠️ Cleaning Decisions (Person 2)")
decisions = {k: st.sidebar.selectbox(k, v) for k, v in P.DECISIONS.items()}

log = DecisionLog()
df_clean = P.clean(decisions, log)
res = P.run_pipeline(decisions)
stats_res = P.run_stats(df_clean)

# Metrics
m1, m2, m3 = st.columns(3)
m1.metric("E1: Disengagement Withdrawal Gap", f"{res['E1'] * 100:.1f}%", help="Withdrawal rate difference between bottom and top activity quartiles")
m2.metric("E2: Early Assessment Score Gap", f"{res['E2']:.1f} pts", help="Average early assignment score difference between persisting and withdrawn students")
m3.metric("E3: Click Drop-Off Severity", f"{res['E3'] * 100:.1f}%", help="Difference in relative week-1 to cutoff activity dropoff")

st.markdown("---")
col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.subheader("Weekly Click Activity Trajectories (L5)")
    st.write("Disengaged learners display a sharp drop-off in interaction as early as week 2–3.")
    week_cols = [f"week_{w}" for w in range(9) if f"week_{w}" in df_clean.columns]
    persist_traj = df_clean.loc[df_clean["withdrawn"] == 0, week_cols].mean()
    withdrawn_traj = df_clean.loc[df_clean["withdrawn"] == 1, week_cols].mean()

    fig, ax = plt.subplots(figsize=(6, 4))
    weeks = list(range(len(week_cols)))
    ax.plot(weeks, persist_traj.values, marker="o", label="Persisted / Passed", color=PULSE_COLOURS["engagement"], lw=2)
    ax.plot(weeks, withdrawn_traj.values, marker="s", label="Withdrawn", color="#e53e3e", lw=2, linestyle="--")
    badge(ax, "OULAD (UK)", real=True)
    ax.set_title("Mean Weekly Clicks: Persisted vs Withdrawn")
    ax.set_xlabel("Course Week")
    ax.set_ylabel("Mean Weekly Clicks")
    ax.set_xticks(weeks)
    ax.set_xticklabels([f"Wk {w}" for w in weeks])
    ax.legend()
    st.pyplot(fig)

with col_chart2:
    st.subheader("Early Assignment Performance (L3–4)")
    st.write("Early continuous assessment scores offer an objective academic cross-validation to click telemetry.")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.kdeplot(data=df_clean, x="early_assessment_score", hue="final_result", common_norm=False, ax=ax2, fill=True)
    badge(ax2, "OULAD (UK)", real=True)
    ax2.set_title("Early Assessment Score Distribution")
    ax2.set_xlabel("Assignment Score (0–100)")
    st.pyplot(fig2)

st.markdown("---")
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Hypothesis Tests (χ², t, z)",
    "📉 Trajectory PCA & t-SNE",
    "👥 Learner Trajectory Clustering",
    "🔍 Association Rules Mining",
    "📝 Decision Log",
    "🔬 Finding Robustness Matrix"
])

with tab1:
    st.markdown("### Hypothesis Tests & Effect Sizes")
    stats_data = [
        {"Predictor": "Low Early Clicks (E1: χ²)", "Test": "Chi-Square Test", "Statistic": f"Chi2 = {stats_res['clicks_chi2']['chi2']:.1f}", "p-value": f"{stats_res['clicks_chi2']['p']:.2e}", "Effect Size": f"Cramér's V = {stats_res['clicks_chi2']['cramers_v']:.3f}"},
        {"Predictor": "Low Early Clicks (E1: z-test)", "Test": "Two-Proportion z-test", "Statistic": f"z = {stats_res['clicks_z']['z']:.2f}", "p-value": f"{stats_res['clicks_z']['p']:.2e}", "Effect Size": f"Diff = +{stats_res['clicks_z']['diff']*100:.1f}%"},
        {"Predictor": "Early Assessment Score (E2: t-test)", "Test": "Welch's Two-Sample t-test", "Statistic": f"t = {stats_res['scores_t']['t']:.2f}", "p-value": f"{stats_res['scores_t']['p']:.2e}", "Effect Size": f"Cohen's d = {stats_res['scores_t']['cohens_d']:.3f}"},
    ]
    st.dataframe(pd.DataFrame(stats_data), use_container_width=True)

with tab2:
    st.markdown("### Trajectory Dimensionality Reduction: PCA & t-SNE (L12–14)")
    decomp = P.run_decomposition(df_clean)
    c1, c2 = st.columns(2)
    with c1:
        st.write(f"**PCA of Weekly Click Curves (Exp. Var: {sum(decomp['pca_variance']):.1%}):**")
        fig_pca, ax_pca = plt.subplots(figsize=(6, 4))
        ax_pca.scatter(decomp["pca_pts"][:, 0], decomp["pca_pts"][:, 1], c=df_clean["withdrawn"], cmap="bwr", alpha=0.3, s=10)
        badge(ax_pca, "OULAD (UK)", real=True)
        st.pyplot(fig_pca)
    with c2:
        st.write("**t-SNE Embedding of Engagement Trajectories:**")
        fig_tsne, ax_tsne = plt.subplots(figsize=(6, 4))
        sample_w = df_clean.loc[decomp["sample_idx"], "withdrawn"]
        ax_tsne.scatter(decomp["tsne_pts"][:, 0], decomp["tsne_pts"][:, 1], c=sample_w, cmap="bwr", alpha=0.7, s=20)
        badge(ax_tsne, "OULAD (UK)", real=True)
        st.pyplot(fig_tsne)

with tab3:
    st.markdown("### Clustering Trajectories into Learner Archetypes (L19–21)")
    metrics, profiles = P.run_clustering(df_clean, k=4)
    st.write("**Clustering Algorithms Evaluation (KMeans vs Agglo vs DBSCAN):**")
    st.dataframe(metrics, use_container_width=True)
    st.write("**Learner Trajectory Archetypes:**")
    st.dataframe(profiles, use_container_width=True)

with tab4:
    st.markdown("### Apriori Association Rules targeting Withdrawal (L18)")
    rules_df = P.run_rules(df_clean)
    st.dataframe(rules_df, use_container_width=True)

with tab5:
    st.markdown("### Auditable Data-Cleaning Decision Log")
    st.dataframe(log.to_frame(), use_container_width=True)

with tab6:
    st.markdown("### Finding Robustness Matrix")
    v, s = robustness_matrix(P.run_pipeline, P.DECISIONS, P.FINDINGS)
    st.pyplot(heatmap(v, s, "Engagement Pulse: Finding Robustness Heatmap"))
    st.caption(f"**Robustness Score:** {int(s.values.sum())} / {s.size} variants survive.")
