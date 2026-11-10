import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from campuspulse.common.plots import theme, badge, PULSE_COLOURS

theme()

st.set_page_config(page_title="Campus Calendar", page_icon="📅", layout="wide")
st.title("📅 ⑥ Campus Calendar: Unified Semester Timeline")
st.caption("How the academic calendar synchronizes the five vital signs across the semester lifecycle")

st.markdown("""
A university campus pulses according to a shared institutional clock. While each dataset originates from a different institution,
their operational signals map directly onto the universal **16-week academic semester timeline**.
""")

# Timeline Data across 16 weeks
semester_events = [
    {"Pulse": "① Academic", "Start": 0, "End": 2, "Event": "Enrolment & Fee Registration", "Color": PULSE_COLOURS["academic"]},
    {"Pulse": "① Academic", "Start": 3, "End": 5, "Event": "Tuition Debt Cutoff (A1)", "Color": PULSE_COLOURS["academic"]},
    {"Pulse": "① Academic", "Start": 7, "End": 9, "Event": "Midterm Progress Reviews", "Color": PULSE_COLOURS["academic"]},
    {"Pulse": "① Academic", "Start": 14, "End": 16, "Event": "Final Exams & Grade Filing (A3)", "Color": PULSE_COLOURS["academic"]},

    {"Pulse": "② Engagement", "Start": 0, "End": 1, "Event": "Platform Orientation", "Color": PULSE_COLOURS["engagement"]},
    {"Pulse": "② Engagement", "Start": 2, "End": 4, "Event": "Early Warning Window (E1: Clicks Drop)", "Color": PULSE_COLOURS["engagement"]},
    {"Pulse": "② Engagement", "Start": 4, "End": 6, "Event": "First Assignment Deadline (E2)", "Color": PULSE_COLOURS["engagement"]},
    {"Pulse": "② Engagement", "Start": 7, "End": 10, "Event": "Formal Withdrawal Peak", "Color": PULSE_COLOURS["engagement"]},

    {"Pulse": "③ Wellbeing", "Start": 1, "End": 3, "Event": "Orientation & Social Adjustment", "Color": PULSE_COLOURS["wellbeing"]},
    {"Pulse": "③ Wellbeing", "Start": 7, "End": 9, "Event": "Midterm Stress & Sleep Dip (W1)", "Color": PULSE_COLOURS["wellbeing"]},
    {"Pulse": "③ Wellbeing", "Start": 13, "End": 16, "Event": "Endsem Acute Stress & Sleep Crisis", "Color": PULSE_COLOURS["wellbeing"]},

    {"Pulse": "④ Food", "Start": 1, "End": 3, "Event": "Basis Cycle 1: Baseline Meat/Veg (F1)", "Color": PULSE_COLOURS["food"]},
    {"Pulse": "④ Food", "Start": 3, "End": 7, "Event": "Intervention 1: Veg Prominence (+20% F2)", "Color": PULSE_COLOURS["food"]},
    {"Pulse": "④ Food", "Start": 7, "End": 9, "Event": "Basis Cycle 2: Midterm Baseline", "Color": PULSE_COLOURS["food"]},
    {"Pulse": "④ Food", "Start": 9, "End": 13, "Event": "Intervention 2: Plant-Based Buffet", "Color": PULSE_COLOURS["food"]},

    {"Pulse": "⑤ Energy", "Start": 1, "End": 7, "Event": "Standard Academic Load", "Color": PULSE_COLOURS["energy"]},
    {"Pulse": "⑤ Energy", "Start": 7, "End": 9, "Event": "Midterm Late-Night Hostel Surge (+2 kW EN1)", "Color": PULSE_COLOURS["energy"]},
    {"Pulse": "⑤ Energy", "Start": 10, "End": 12, "Event": "Campus Tech Fest / Cultural Break", "Color": PULSE_COLOURS["energy"]},
    {"Pulse": "⑤ Energy", "Start": 14, "End": 16, "Event": "Endsem Late-Night Hostel Surge (EN1)", "Color": PULSE_COLOURS["energy"]},
]

# Plot timeline chart
st.subheader("Visualizing the Five Vital Signs on One Semester Timeline")
fig, ax = plt.subplots(figsize=(12, 5))

pulses_list = ["① Academic", "② Engagement", "③ Wellbeing", "④ Food", "⑤ Energy"]
y_pos = {p: i for i, p in enumerate(pulses_list)}

for ev in semester_events:
    y = y_pos[ev["Pulse"]]
    start = ev["Start"]
    width = ev["End"] - ev["Start"]
    ax.barh(y, width, left=start, height=0.5, color=ev["Color"], alpha=0.85, edgecolor="#2d3748")
    ax.text(start + width / 2.0, y, ev["Event"], ha="center", va="center", color="white", fontsize=7.5, weight="bold")

ax.set_yticks(range(len(pulses_list)))
ax.set_yticklabels(pulses_list, fontsize=10, weight="bold")
ax.set_xlabel("Semester Timeline (Weeks 0 to 16)", fontsize=10)
ax.set_xlim(-0.5, 16.5)
ax.set_xticks(range(17))
ax.set_xticklabels([f"Wk {w}" for w in range(17)])
ax.grid(axis="x", linestyle="--", alpha=0.5)
ax.set_title("CampusPulse Semester Lifecycle Synchronization", fontsize=12)
badge(ax, "CampusPulse Multi-Campus Timeline", real=True)

st.pyplot(fig)

st.markdown("---")
st.subheader("Interactive Weekly Vital Sign Inspector")

selected_week = st.slider("Select Semester Week to Inspect Vital Signals:", min_value=0, max_value=16, value=8)

w_cols = st.columns(5)
vital_status = {
    "Academic": "Normal instruction" if selected_week not in [4, 8, 14, 15, 16] else ("⚠️ Tuition Payment Default Warning!" if selected_week == 4 else "📝 Exam Evaluations"),
    "Engagement": "Routine online clicks" if selected_week not in [2, 3, 4, 5] else "🚨 Disengagement Detection Window (E1) — Action Required!",
    "Wellbeing": "Baseline mental health" if selected_week not in [8, 14, 15, 16] else "🚨 Acute Sleep Deprivation & Stress Surge (W1)",
    "Food": "Baseline Canteen Operation" if selected_week in [1, 2, 7, 8] else "🥦 Vegetarian Intervention Week (Higher Plant Selection)",
    "Energy": "Normal Building Power Profile" if selected_week not in [8, 14, 15, 16] else "⚡ Hostel Late-Night Power Surge (>2 kW EN1)",
}

for col, (pulse_name, status) in zip(w_cols, vital_status.items()):
    with col:
        st.markdown(f"**{pulse_name}**")
        if "🚨" in status or "⚠️" in status:
            st.error(status)
        elif "🥦" in status or "⚡" in status or "📝" in status:
            st.warning(status)
        else:
            st.success(status)

st.markdown("---")
st.subheader("Cross-Pulse Syntheses")
c1, c2 = st.columns(2)
with c1:
    st.markdown("""
    #### 🔗 Academic ① ↔ Engagement ②: Institutional Transferability
    - **Physical Campus (Portugal):** Academic dropout is heavily signaled by financial hardship (Debtor status: +41.1% dropout rate) and first-semester unsubmitted exams.
    - **Online University (UK):** Disengagement occurs earlier and silently; by Week 4, a student in the bottom activity quartile is +45.4% more likely to drop out.
    - **Synthesis:** In both physical and digital institutions, early non-engagement precedes administrative departure by 4–8 weeks, giving the university an actionable window.
    """)

with c2:
    st.markdown("""
    #### 🔗 Wellbeing ③ ↔ Energy ⑤: Sleep Deprivation & Campus Power
    - **YRBS Survey:** Sleep duration exhibits a monotonic link with persistent sadness (47.9% for <7 hrs vs 19.1% for ≥8 hrs).
    - **I-BLEND Sensors:** During exam weeks (Wk 8 & Wk 15), late-night (00:00–05:00) hostel electricity surges by +2,151 Watts as students study overnight.
    - **Synthesis:** The hostel power grid serves as a non-intrusive, aggregate sensor for campus-wide sleep disruption and exam anxiety.
    """)
