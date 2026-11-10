import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
from campuspulse.common.dashboard import PULSES, load_pulse, is_ready

st.set_page_config(
    page_title="CampusPulse — Campus Health Report",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🩺 CampusPulse: Campus Health Report")
st.subheader("Five Vital Signs of a Campus Under One Shared Methodology")

st.markdown("""
> **The Umbrella Question:**
> *Can routine campus data (records, clicks, surveys, sales, meters) warn a university early enough to act,
> and how much do those warnings depend on our data-cleaning decisions?*
""")

st.write("""
Just as a physician assesses vital signs to evaluate patient stability, **CampusPulse** monitors the five vital signs
of a modern university campus—spanning both its **People** (academics, engagement, mental health) and its **Place** (food choices, building energy).

Every vital sign is analyzed using real public institutional datasets and governed by a shared contract:
every pipeline exposes an auditable **Decision Log** and a **Finding Robustness Matrix** to test whether insights
survive alternative data-cleaning decisions.
""")

st.markdown("---")
st.subheader("Campus Vital Signs Overview")

cols = st.columns(5)
vital_data = [
    {
        "name": "academic",
        "title": "① Academic Pulse",
        "question": "Who drops out? How early can we tell?",
        "dataset": "UCI 697 Dropout (4,424 students)",
        "key_metric": "Debtor Dropout Surge",
        "metric_val": "+41.1%",
        "lead": "Person 1",
        "link": "1_Academic_Pulse"
    },
    {
        "name": "engagement",
        "title": "② Engagement Pulse",
        "question": "When do online learners go quiet?",
        "dataset": "OULAD UK (32,593 students, 10.6M clicks)",
        "key_metric": "Disengagement Withdrawal Gap",
        "metric_val": "+45.4%",
        "lead": "Person 2",
        "link": "2_Engagement_Pulse"
    },
    {
        "name": "wellbeing",
        "title": "③ Wellbeing Pulse",
        "question": "What co-occurs with persistent sadness?",
        "dataset": "CDC YRBS (10.7k) + Malaysia MH (101)",
        "key_metric": "Sleep Deprivation Risk",
        "metric_val": "+28.0%",
        "lead": "Person 3",
        "link": "3_Wellbeing_Pulse"
    },
    {
        "name": "food",
        "title": "④ Food Pulse",
        "question": "Do menus change what students choose?",
        "dataset": "NOVANIMAL ZHAW (26.3k menu sales)",
        "key_metric": "Veg Share Jump",
        "metric_val": "+19.3%",
        "lead": "Person 4",
        "link": "4_Food_Pulse"
    },
    {
        "name": "energy",
        "title": "⑤ Energy Pulse",
        "question": "When does a building behave unusually?",
        "dataset": "I-BLEND IIIT Delhi (52 mo. meters)",
        "key_metric": "Exam Late-Night Surge",
        "metric_val": "+2.15 kW",
        "lead": "Person 5",
        "link": "5_Energy_Pulse"
    },
]

for c, v in zip(cols, vital_data):
    mod = load_pulse(v["name"])
    ready = is_ready(mod)
    with c:
        st.markdown(f"### {v['title']}")
        st.caption(f"**Lead:** {v['lead']} | **Status:** {'🟢 Ready' if ready else '🟡 Pending'}")
        st.info(f"**Problem:** {v['question']}")
        st.metric(label=v["key_metric"], value=v["metric_val"])
        st.caption(f"**Dataset:** {v['dataset']}")

st.markdown("---")
st.subheader("Core Architectural Links Across Pulses")

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("""
    #### 📅 Shared Semester Timeline
    All five pulses align with the university academic calendar:
    - **Academic:** Semester 1 & 2 course enrollment and grade posting milestones.
    - **Engagement:** Weeks -1 to 8 platform activity curves and assignment deadlines.
    - **Food:** 12-week canteen field experiment alternating between Basis and Vegetarian Intervention cycles.
    - **Energy:** Multi-year institute calendar capturing midterm and endterm exam seasons.
    
    👉 Explore the interactive **[Campus Calendar](pages/6_Campus_Calendar)** to see all five synchronized.
    """)

with col_right:
    st.markdown("""
    #### 🔬 Finding Robustness Matrix
    *Does the finding survive a different cleaning choice?*
    - Traditional reporting assumes data cleaning choices are objective.
    - CampusPulse tests every claim under alternative cleaning branches (e.g. dropping vs imputing missing grades, smoothing clicks, filtering holidays, clipping spikes).
    - If an effect flips sign or drops below 50% magnitude, it is flagged as **fragile**.
    
    👉 Visit **[How Fragile Are Insights?](pages/7_How_Fragile_Are_Insights)** to inspect the live sensitivity matrices.
    """)
