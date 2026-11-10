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

from campuspulse.wellbeing import pipeline as P
from campuspulse.common.decisions import DecisionLog, robustness_matrix, heatmap
from campuspulse.common.plots import theme, badge, PULSE_COLOURS

theme()

st.set_page_config(page_title="Wellbeing Pulse", page_icon="🧠", layout="wide")
st.title("🧠 ③ Wellbeing Pulse: Student Mental Health & Behavioral Risk Factors")
st.caption("CDC YRBS (10,710 high school / university entrants) + Malaysia Public University MH Survey (101 students)")

st.sidebar.header("🛠️ Cleaning Decisions (Person 3)")
decisions = {k: st.sidebar.selectbox(k, v) for k, v in P.DECISIONS.items()}

log = DecisionLog()
df_yrbs, df_mh = P.clean(decisions, log)
res = P.run_pipeline(decisions)
stats_res = P.run_stats(df_yrbs)

# Metrics
m1, m2, m3 = st.columns(3)
m1.metric("W1: Sleep Deprivation Sadness Gap", f"+{res['W1'] * 100:.1f}%", help="Excess sadness rate among sleep-deprived vs well-rested students")
m2.metric("W2: Bullying Risk Surge", f"+{res['W2'] * 100:.1f}%", help="Excess sadness rate among bullied students")
m3.metric("W3: Inactivity Vulnerability", f"+{res['W3'] * 100:.1f}%", help="Excess sadness rate among physically inactive students")

st.markdown("---")
col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.subheader("Sleep Hours vs Persistent Sadness Gradient (L3–4)")
    st.write("A monotonic relationship: students sleeping ≤4 hours experience over 3.8× the rate of persistent sadness compared to those sleeping ≥8 hours.")
    
    sleep_labels = {1: "≥8 hrs", 2: "7 hrs", 3: "6 hrs", 4: "5 hrs", 5: "≤4 hrs"}
    sleep_agg = df_yrbs.groupby("q88")["sadness"].mean().reset_index()
    sleep_agg = sleep_agg[sleep_agg["q88"].isin(sleep_labels.keys())]
    sleep_agg["Label"] = sleep_agg["q88"].map(sleep_labels)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=sleep_agg, x="Label", y="sadness", ax=ax, color=PULSE_COLOURS["wellbeing"])
    badge(ax, "CDC YRBS", real=True)
    ax.set_ylabel("Rate of Persistent Sadness")
    ax.set_xlabel("Hours of Nightly Sleep")
    ax.set_title("Persistent Sadness by Reported Sleep Duration")
    for p in ax.patches:
        ax.annotate(f"{p.get_height()*100:.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha="center", va="bottom", fontsize=9, xytext=(0, 2), textcoords="offset points")
    st.pyplot(fig)

with col_chart2:
    st.subheader("Bullying & Inactivity Impact Comparison")
    st.write("Comparing persistent sadness rates across behavioral vulnerabilities.")
    
    comp_df = pd.DataFrame([
        {"Factor": "Sleep Rested", "Category": "Sleep", "Sadness": df_yrbs.loc[df_yrbs["sleep_deprived"] == 0, "sadness"].mean()},
        {"Factor": "Sleep Deprived", "Category": "Sleep", "Sadness": df_yrbs.loc[df_yrbs["sleep_deprived"] == 1, "sadness"].mean()},
        {"Factor": "Not Bullied", "Category": "Bullying", "Sadness": df_yrbs.loc[df_yrbs["bullied"] == 0, "sadness"].mean()},
        {"Factor": "Bullied", "Category": "Bullying", "Sadness": df_yrbs.loc[df_yrbs["bullied"] == 1, "sadness"].mean()},
        {"Factor": "Physically Active", "Category": "Activity", "Sadness": df_yrbs.loc[df_yrbs["inactive"] == 0, "sadness"].mean()},
        {"Factor": "Inactive", "Category": "Activity", "Sadness": df_yrbs.loc[df_yrbs["inactive"] == 1, "sadness"].mean()},
    ])
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.barplot(data=comp_df, x="Factor", y="Sadness", hue="Category", ax=ax2, dodge=False)
    badge(ax2, "CDC YRBS", real=True)
    ax2.set_ylabel("Persistent Sadness Rate")
    ax2.set_title("Vulnerability Profiles")
    plt.xticks(rotation=25, ha="right")
    st.pyplot(fig2)

st.markdown("---")
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📝 Course Normalization (L9)",
    "📉 Behavioral PCA & t-SNE",
    "👥 Risk Profile Clustering",
    "🔍 Association Rules Mining (L18)",
    "📊 Hypothesis Tests & Bonferroni",
    "🔬 Finding Robustness Matrix"
])

with tab1:
    st.subheader("Signature Challenge: Malaysia Survey Free-Text Course Normalization (L9)")
    st.write("The 101-student survey featured 49 inconsistent spellings for academic courses. Fuzzy matching groups them into 7 standardized disciplines:")
    if not df_mh.empty and "course_clean" in df_mh.columns:
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Discipline Counts & Depression Prevalence:**")
            mh_summary = df_mh.groupby("course_clean").agg(
                Students=("Timestamp", "count"),
                Depression_Rate=("Do you have Depression?", lambda s: (s == "Yes").mean() * 100)
            ).reset_index()
            st.dataframe(mh_summary.sort_values("Students", ascending=False), use_container_width=True)
        with c2:
            st.write("**Sample Fuzzy Normalizations:**")
            sample_mappings = [
                {"Raw User Input": "Engine / koe / KOE / engin", "Normalized Faculty": "Engineering"},
                {"Raw User Input": "BIT / BCS / IT / CTS", "Normalized Faculty": "Computing & IT"},
                {"Raw User Input": "Laws / Law", "Normalized Faculty": "Law"},
                {"Raw User Input": "Pendidikan islam / Irkhs / Usuluddin", "Normalized Faculty": "Islamic Studies"},
                {"Raw User Input": "Accounting / Banking / Econs", "Normalized Faculty": "Business & Economics"}
            ]
            st.dataframe(pd.DataFrame(sample_mappings), use_container_width=True)

with tab2:
    st.subheader("Behavioral Risk Factor PCA & t-SNE (L12–14)")
    decomp = P.run_decomposition(df_yrbs)
    c1, c2 = st.columns(2)
    with c1:
        st.write(f"**PCA Projection (Explained Var: {sum(decomp['pca_variance']):.1%}):**")
        fig_pca, ax_pca = plt.subplots(figsize=(6, 4))
        ax_pca.scatter(decomp["pca_pts"][:, 0], decomp["pca_pts"][:, 1], c=df_yrbs["sadness"], cmap="coolwarm", alpha=0.3, s=10)
        badge(ax_pca, "CDC YRBS", real=True)
        st.pyplot(fig_pca)
    with c2:
        st.write("**t-SNE Embedding of Risk Behaviors:**")
        fig_tsne, ax_tsne = plt.subplots(figsize=(6, 4))
        sample_s = df_yrbs.loc[decomp["sample_idx"], "sadness"]
        ax_tsne.scatter(decomp["tsne_pts"][:, 0], decomp["tsne_pts"][:, 1], c=sample_s, cmap="coolwarm", alpha=0.7, s=20)
        badge(ax_tsne, "CDC YRBS", real=True)
        st.pyplot(fig_tsne)

with tab3:
    st.subheader("Clustering Behavioral Risk Personas (L19–21)")
    metrics, profiles = P.run_clustering(df_yrbs, k=4)
    st.write("**Clustering Algorithms Comparison (KMeans vs Agglomerative vs DBSCAN):**")
    st.dataframe(metrics, use_container_width=True)
    st.write("**Behavioral Risk Personas:**")
    st.dataframe(profiles, use_container_width=True)

with tab4:
    st.subheader("Apriori Association Rules Mining (L18)")
    rules_df = P.mine_sadness_rules(df_yrbs)
    st.dataframe(rules_df, use_container_width=True)

with tab5:
    st.subheader("Statistical Tests with Bonferroni Adjustment")
    stats_data = [
        {"Factor": "Sleep Deprivation (W1)", "Test": "Chi-Square", "Statistic": f"Chi2 = {stats_res['sleep_chi2']['chi2']:.1f}", "p-value": f"{stats_res['sleep_chi2']['p']:.2e}", "Effect Size": f"Cramér's V = {stats_res['sleep_chi2']['cramers_v']:.3f}"},
        {"Factor": "Bullying Victimization (W2)", "Test": "Chi-Square", "Statistic": f"Chi2 = {stats_res['bully_chi2']['chi2']:.1f}", "p-value": f"{stats_res['bully_chi2']['p']:.2e}", "Effect Size": f"Cramér's V = {stats_res['bully_chi2']['cramers_v']:.3f}"},
        {"Factor": "Physical Inactivity (W3)", "Test": "Chi-Square", "Statistic": f"Chi2 = {stats_res['inactivity_chi2']['chi2']:.1f}", "p-value": f"{stats_res['inactivity_chi2']['p']:.2e}", "Effect Size": f"Cramér's V = {stats_res['inactivity_chi2']['cramers_v']:.3f}"},
    ]
    st.dataframe(pd.DataFrame(stats_data), use_container_width=True)
    st.write("**Bonferroni Multiple Testing Adjustment:**")
    st.dataframe(stats_res["bonferroni"], use_container_width=True)

with tab6:
    st.subheader("Finding Robustness Matrix")
    v, s = robustness_matrix(P.run_pipeline, P.DECISIONS, P.FINDINGS)
    st.pyplot(heatmap(v, s, "Wellbeing Pulse: Finding Robustness Heatmap"))
    st.caption(f"**Robustness Score:** {int(s.values.sum())} / {s.size} variants survive.")
