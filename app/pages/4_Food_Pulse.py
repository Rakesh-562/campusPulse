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

from campuspulse.food import pipeline as P
from campuspulse.common.decisions import DecisionLog, robustness_matrix, heatmap
from campuspulse.common.plots import theme, badge, PULSE_COLOURS

theme()

st.set_page_config(page_title="Food Pulse", page_icon="🥗", layout="wide")
st.title("🥗 ④ Food Pulse: Canteen Menu Choice & Vegetarian Field Experiment")
st.caption("NOVANIMAL (ZHAW Zurich University of Applied Sciences, Switzerland) — 26,340 transaction records across 2 canteens")

st.sidebar.header("🛠️ Cleaning Decisions (Person 4)")
decisions = {k: st.sidebar.selectbox(k, v) for k, v in P.DECISIONS.items()}

log = DecisionLog()
df_clean = P.clean(decisions, log)
res = P.run_pipeline(decisions)
stats_res = P.run_stats(df_clean)

# Metrics
m1, m2, m3 = st.columns(3)
m1.metric("F1: Daily Volume Stability", f"+{res['F1']:.1f} meals", help="Daily sales non-inferiority between Intervention and Basis cycles")
m2.metric("F2: Vegetarian Share Surge", f"+{res['F2'] * 100:.1f}%", help="Increase in vegetarian/vegan meal selection share during intervention weeks")
m3.metric("F3: Student vs Staff Choice Gap", f"+{res['F3'] * 100:.1f}%", help="Higher baseline vegetarian preference among students compared to staff")

st.markdown("---")
col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.subheader("Total Daily Canteen Sales: Basis vs Intervention (L6–7)")
    st.write("Does offering more vegetarian meals harm cafeteria turnover? The randomized field trial proves sales volume is fully preserved.")
    
    daily_sales = df_clean.groupby(["date", "condit"])["qty_weight"].sum().reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=daily_sales, x="condit", y="qty_weight", ax=ax, palette=["#3182ce", "#38a169"])
    badge(ax, "NOVANIMAL (ZHAW)", real=True)
    ax.set_ylabel("Total Daily Meals Sold")
    ax.set_xlabel("Experiment Cycle")
    ax.set_title("Canteen Daily Volume Distribution")
    st.pyplot(fig)

with col_chart2:
    st.subheader("Vegetarian & Plant-Based Choice Share (L5)")
    st.write("Proportion of meals selected from vegetarian or vegan offerings jumped from 27.6% in Basis to 47.1% in Intervention.")
    
    daily_veg = df_clean.groupby(["date", "condit"]).apply(
        lambda g: (g["is_veg"] * g["qty_weight"]).sum() / g["qty_weight"].sum()
    ).reset_index(name="veg_share")
    
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.kdeplot(data=daily_veg, x="veg_share", hue="condit", fill=True, ax=ax2, palette=["#3182ce", "#38a169"])
    badge(ax2, "NOVANIMAL (ZHAW)", real=True)
    ax2.set_xlabel("Daily Vegetarian Meal Share")
    ax2.set_title("Shift in Consumer Vegetarian Choice Distribution")
    st.pyplot(fig2)

st.markdown("---")
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "👥 Customer Segment Response",
    "📉 Menu Demand PCA (L12–13)",
    "🍲 Daily Demand Clustering",
    "🔍 Association Rules Mining (L18)",
    "📝 Decision Log",
    "🔬 Finding Robustness Matrix"
])

with tab1:
    st.subheader("Customer Segment Differences: Students vs Staff")
    st.write("Comparing the response of university students (`Studierende`) vs staff members (`Mitarbeiter`):")
    raw_df = P.load_raw()
    seg_summary = []
    for cond in ["Basis", "Intervention"]:
        sub = raw_df[raw_df["condit"] == cond]
        for grp in ["Studierende", "Mitarbeiter"]:
            sub_g = sub[sub["rab_descript"] == grp]
            tot = sub_g["qty_weight"].sum()
            veg = (sub_g["is_veg"] * sub_g["qty_weight"]).sum()
            seg_summary.append({"Cycle": cond, "Segment": grp, "Total_Meals": int(tot), "Veg_Share": f"{veg/tot*100:.1f}%"})
    st.dataframe(pd.DataFrame(seg_summary), use_container_width=True)

with tab2:
    st.subheader("PCA of Daily Menu Choices (L12–13)")
    decomp = P.run_decomposition(df_clean)
    fig_pca, ax_pca = plt.subplots(figsize=(7, 4))
    ax_pca.scatter(decomp["pca_pts"][:, 0], decomp["pca_pts"][:, 1], color="#dd6b20", alpha=0.7, s=40)
    ax_pca.set_title(f"PCA of Canteen Demand (Exp. Var: {sum(decomp['pca_variance']):.1%})")
    ax_pca.set_xlabel("PC1 (Meat vs Veg Volume)")
    ax_pca.set_ylabel("PC2 (Buffet vs Plated)")
    badge(ax_pca, "NOVANIMAL", real=True)
    st.pyplot(fig_pca)

with tab3:
    st.subheader("Clustering Canteen Demand Days (L19–21)")
    metrics, profiles = P.run_clustering(df_clean, k=3)
    st.write("**Clustering Algorithms Comparison (KMeans vs Agglomerative vs DBSCAN):**")
    st.dataframe(metrics, use_container_width=True)
    st.write("**Canteen Operating Day Profiles (Mean Sales per Category):**")
    st.dataframe(profiles, use_container_width=True)

with tab4:
    st.subheader("Apriori Association Rules (L18)")
    rules_df = P.mine_food_rules(df_clean)
    st.dataframe(rules_df, use_container_width=True)

with tab5:
    st.subheader("Auditable Data-Cleaning Decision Log")
    st.dataframe(log.to_frame(), use_container_width=True)

with tab6:
    st.subheader("Finding Robustness Matrix")
    v, s = robustness_matrix(P.run_pipeline, P.DECISIONS, P.FINDINGS)
    st.pyplot(heatmap(v, s, "Food Pulse: Finding Robustness Heatmap"))
    st.caption(f"**Robustness Score:** {int(s.values.sum())} / {s.size} variants survive.")
