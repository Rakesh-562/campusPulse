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

from campuspulse.academic import pipeline as P
from campuspulse.common.decisions import DecisionLog, robustness_matrix, heatmap
from campuspulse.common.plots import theme, badge, PULSE_COLOURS

theme()

st.set_page_config(page_title="Academic Pulse", page_icon="🎓", layout="wide")
st.title("🎓 ① Academic Pulse: Dropout Early Warning")
st.caption("UCI 697 Predict Students' Dropout and Academic Success (4,424 students, 37 columns)")

st.sidebar.header("🛠️ Cleaning Decisions (Person 1)")
decisions = {k: st.sidebar.selectbox(k, v) for k, v in P.DECISIONS.items()}

# Run pipeline with decision log
log = DecisionLog()
df_clean = P.clean(decisions, log)
res = P.run_pipeline(decisions)
stats_res = P.run_stats(df_clean)

# Metrics
m1, m2, m3 = st.columns(3)
m1.metric("A1: Debtor Dropout Gap", f"{res['A1'] * 100:.1f}%", help="Difference in dropout rate between debtors and non-debtors")
m2.metric("A2: Scholarship Protection", f"{res['A2'] * 100:.1f}%", help="Reduction in dropout rate for scholarship recipients")
m3.metric("A3: Grade Gap (Persist vs Dropout)", f"{res['A3']:.2f} pts", help="Average 1st semester grade difference")

st.markdown("---")
col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.subheader("Signature Challenge: The Hidden Zeros (L7–8)")
    st.write("538 students scored exactly 0.0 in Semester 1. Over 92% of them dropped out. Are they genuine academic failures, or unsubmitted/missing records?")
    fig, ax = plt.subplots(figsize=(6, 4))
    raw_df = P.load_raw()
    sns.histplot(data=raw_df, x=P.GRADE, hue="Target", bins=20, multiple="stack", ax=ax, palette="tab10")
    badge(ax, "UCI Dropout", real=True)
    ax.set_title("Semester 1 Grade Distribution by Outcome")
    ax.set_xlabel("1st Semester Grade (0 to 20 scale)")
    st.pyplot(fig)

with col_chart2:
    st.subheader("Financial Hardship vs Academic Support")
    st.write("Tuition debt is the single strongest pre-academic predictor of student withdrawal.")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    debt_df = df_clean.groupby(["Debtor", "Scholarship holder"])["dropout"].mean().reset_index()
    debt_df["Label"] = debt_df.apply(lambda r: f"Debt={r['Debtor']}, Schol={r['Scholarship holder']}", axis=1)
    sns.barplot(data=debt_df, x="Label", y="dropout", ax=ax2, color=PULSE_COLOURS["academic"])
    badge(ax2, "UCI Dropout", real=True)
    ax2.set_ylabel("Dropout Rate")
    ax2.set_title("Dropout Rate by Debt & Scholarship Status")
    plt.xticks(rotation=20, ha="right")
    st.pyplot(fig2)

st.markdown("---")
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Hypothesis Tests (χ², t, F, z)",
    "📉 Dimensionality Reduction (PCA & t-SNE)",
    "👥 Academic Personas Clustering",
    "🔍 Association Rules Mining",
    "📝 Decision Log",
    "🔬 Finding Robustness Matrix"
])

with tab1:
    st.markdown("### Hypothesis Tests & Effect Sizes (L3–4)")
    stats_data = [
        {"Predictor / Test": "Tuition Debtor (A1: χ²)", "Test Type": "Chi-Square Contingency", "Statistic": f"Chi2 = {stats_res['debtor_chi2']['chi2']:.1f}", "p-value": f"{stats_res['debtor_chi2']['p']:.2e}", "Effect Size": f"Cramér's V = {stats_res['debtor_chi2']['cramers_v']:.3f}"},
        {"Predictor / Test": "Debtor Dropout (z-test)", "Test Type": "Two-Proportion z-test", "Statistic": f"z = {stats_res['debtor_z']['z']:.2f}", "p-value": f"{stats_res['debtor_z']['p']:.2e}", "Effect Size": f"Diff = +{stats_res['debtor_z']['diff']*100:.1f}%"},
        {"Predictor / Test": "Scholarship Holder (A2: χ²)", "Test Type": "Chi-Square Contingency", "Statistic": f"Chi2 = {stats_res['scholarship_chi2']['chi2']:.1f}", "p-value": f"{stats_res['scholarship_chi2']['p']:.2e}", "Effect Size": f"Cramér's V = {stats_res['scholarship_chi2']['cramers_v']:.3f}"},
        {"Predictor / Test": "Semester 1 Grade (A3: t-test)", "Test Type": "Welch's Two-Sample t-test", "Statistic": f"t = {stats_res['grade_t']['t']:.2f}", "p-value": f"{stats_res['grade_t']['p']:.2e}", "Effect Size": f"Cohen's d = {stats_res['grade_t']['cohens_d']:.3f}"},
        {"Predictor / Test": "Grades Across Courses (ANOVA F)", "Test Type": "One-Way ANOVA", "Statistic": f"F = {stats_res['course_anova']['F']:.1f}", "p-value": f"{stats_res['course_anova']['p']:.2e}", "Effect Size": f"Eta-Squared = {stats_res['course_anova']['eta_sq']:.3f}"},
    ]
    st.dataframe(pd.DataFrame(stats_data), use_container_width=True)

with tab2:
    st.markdown("### Dimensionality Reduction: PCA and t-SNE (L12–14)")
    decomp = P.run_decomposition(df_clean)
    c_p1, c_p2 = st.columns(2)
    with c_p1:
        st.write(f"**PCA Projection (Top 2 Components, Var: {sum(decomp['pca_variance']):.1%}):**")
        fig_pca, ax_pca = plt.subplots(figsize=(6, 4))
        ax_pca.scatter(decomp["pca_pts"][:, 0], decomp["pca_pts"][:, 1], c=df_clean["dropout"], cmap="coolwarm", alpha=0.4, s=15)
        ax_pca.set_xlabel("PC1"); ax_pca.set_ylabel("PC2")
        badge(ax_pca, "UCI Dropout", real=True)
        st.pyplot(fig_pca)
    with c_p2:
        st.write("**t-SNE 2D Manifold Embedding:**")
        fig_tsne, ax_tsne = plt.subplots(figsize=(6, 4))
        sample_dropout = df_clean.loc[decomp["sample_idx"], "dropout"]
        ax_tsne.scatter(decomp["tsne_pts"][:, 0], decomp["tsne_pts"][:, 1], c=sample_dropout, cmap="coolwarm", alpha=0.7, s=20)
        badge(ax_tsne, "UCI Dropout", real=True)
        st.pyplot(fig_tsne)

with tab3:
    st.markdown("### Clustering & Academic Personas (L19–21)")
    metrics, profiles = P.run_clustering(df_clean, k=3)
    st.write("**Clustering Algorithms Comparison (KMeans vs Agglomerative vs DBSCAN):**")
    st.dataframe(metrics, use_container_width=True)
    st.write("**Academic Persona Profiles (Cluster Means):**")
    st.dataframe(profiles, use_container_width=True)

with tab4:
    st.markdown("### Apriori Association Rules Mining targeting Dropout (L18)")
    rules_df = P.run_rules(df_clean)
    st.dataframe(rules_df, use_container_width=True)

with tab5:
    st.markdown("### Auditable Data-Cleaning Decision Log")
    st.dataframe(log.to_frame(), use_container_width=True)

with tab6:
    st.markdown("### Finding Robustness Matrix")
    v, s = robustness_matrix(P.run_pipeline, P.DECISIONS, P.FINDINGS)
    st.pyplot(heatmap(v, s, "Academic Pulse: Finding Robustness Heatmap"))
    st.caption(f"**Robustness Score:** {int(s.values.sum())} / {s.size} variants survive.")
