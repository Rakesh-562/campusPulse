# CampusPulse — Project Plan (v3: five vital signs of a campus)

## Change Log
- **v1**: lifestyle/stress EDA on 3 datasets.
- **v2 (2026-11-09)**: found the Lifestyle dataset is synthetic → pivoted to real dropout data.
- **v3 (2026-11-09)**: five real problems, one per team member, under one theme.

---

## 1. The Idea
A doctor checks a patient's vital signs. **CampusPulse** checks the vital signs of a campus. Each team member owns one, works on one real dataset and one real problem, and contributes one section of a shared dashboard.

```
┌─────────────────────────── CampusPulse ───────────────────────────┐
│              PEOPLE                                PLACE          │
│ ① Academic   ② Engagement   ③ Wellbeing     ④ Food        ⑤ Energy │
│ Who drops    When do online What co-occurs  Do menus      When does│
│ out? How     learners go    with persistent change what   a bldg   │
│ early can    quiet?         sadness?        students      behave   │
│ we tell?                                    choose?       oddly?   │
└───────────────────────────────────────────────────────────────────┘
```

**Umbrella Question:**
*Can routine campus data (records, clicks, surveys, sales, meters) warn a university early enough to act, and how much do those warnings depend on our data-cleaning decisions?*

**Shared Contribution (Everyone):**
Every problem has a decision log and a **Finding Robustness Matrix** (does the finding survive a different cleaning choice?). The dashboard's main page for each pulse lets the viewer flip those choices live. This one method runs through all five problems.

---

## 2. The Five Problems

| # | Pulse | Owner | Real Problem | Dataset (Verified Public) | Signature Data Challenge |
|---|---|---|---|---|---|
| ① | **Academic** | Person 1 | Early warning for university dropout: how early can we tell, and which signals can the university act on (fees, scholarships, semester-1 load)? | **UCI 697** Predict Students' Dropout and Academic Success: 4,424 real students, 37 columns, CC BY 4.0. | Grades of 0 that are really missing values (538 students; 92% dropped out); outcome still unknown for "Enrolled" students; economic columns that really label the cohort |
| ② | **Engagement** | Person 2 | Online-learning disengagement: from which week does a student's learning-platform activity predict withdrawal? | **OULAD** (Open University, UK): 32,593 students, 22 course runs, demographics + assessments + daily platform click logs (`studentVle.csv` ≈ 433 MB). CC BY 4.0 | Very large click logs → aggregate to weeks; noisy, spiky counts (smoothing); missing assessment scores; withdrawal dates; different course runs need aligning |
| ③ | **Wellbeing** | Person 3 | Student mental health: which behaviours (sleep, screen time, bullying, physical activity, substance use) cluster with persistent sadness? | **CDC YRBS** (US Youth Risk Behavior Survey, national sample, public) + **Malaysia university MH survey** (101 rows) | Missing answers on sensitive questions (not random); survey weights; coded answers; the Malaysia survey's 49 spellings of course names (text cleaning / fuzzy matching) |
| ④ | **Food** | Person 4 | Canteen food choice: when the canteen offers more vegetarian meals, do sales drop, and do different customer groups respond differently? | **NOVANIMAL** (ZHAW, Switzerland): 26,000+ aggregated menu sales from 2 university canteens over a 12-week field experiment (2017), base vs intervention weeks. Zenodo 3890931, CC BY 4.0 | Aggregated sales (not individual meals); demand spikes and holidays (outliers vs real events); weekly seasonality (smoothing); the design was randomised, so here we can test cause and effect |
| ⑤ | **Energy** | Person 5 | Campus building behaviour: when does a building's energy use look unusual, and is it a meter error or a real change (exams, holidays, occupancy)? | **I-BLEND** (IIIT Delhi, Indian campus): 7 academic + hostel buildings, 52 months, 1-minute meters, occupancy at 10-min, institute calendar, weather. figshare collection 3893581 | Meter gaps (missing data), spikes (noise vs events), 1-min → hourly/daily resampling, aligning occupancy and calendar data, clustering daily load shapes |

---

## 3. How the Five Connect (The "CampusPulse" Story)

1. **The Academic Calendar runs through all five:**
   - ① Semesters, ② course weeks, ④ base vs intervention weeks, ⑤ IIIT Delhi institute calendar (exams, breaks).
   - The dashboard features a shared **"Campus Calendar"** view marking signals on a semester timeline.

2. **① ↔ ② Cross-institutional early warning:**
   - Compare dropout signals between a campus university (Portugal) and an online one (UK): financial hardship ↔ deprivation band, semester-1 engagement ↔ week-4 clicks, age ↔ age band. Signals that hold in both places establish generalizability.

3. **③ ↔ ⑤ Hostel energy as a campus-wide sign of sleep and stress:**
   - Late-night (00:00–05:00) energy consumption in I-BLEND hostel buildings during exam weeks vs normal weeks. Does the campus "stay up" during exams? Links directly to YRBS sleep findings as an aggregate behavioural proxy.

4. **③ ↔ ④ Food and wellbeing:**
   - YRBS diet patterns (breakfast skipping, vegetables, soda) vs canteen menu choices: what do students choose, and what co-occurs with wellbeing?

5. **Shared Method:**
   - Every pulse reports its own decision log + Robustness Matrix under an identical contract, enabling a side-by-side view: *"How fragile are campus insights?"*

---

## 4. Course Syllabus Coverage

| Topic | ① Dropout | ② OULAD | ③ Wellbeing | ④ Canteen | ⑤ Energy |
|---|:---:|:---:|:---:|:---:|:---:|
| **Collecting/storing (L2)** | ✓ | **Lead**: 433 MB logs → weekly parquet | ✓ survey format conversion | ✓ | **Lead**: 52 months × 1-min → parquet |
| **One/two-variable stats (L3–4)** | ✓ | ✓ | ✓ | ✓ | ✓ |
| **Noise removal / smoothing (L5)** | cohort trend | **Lead**: smooth weekly click curves | — | **Lead**: weekly seasonality smoothing | ✓ load-curve smoothing |
| **Outliers (L6–7)** | age, credits | click spikes | — | demand spikes vs holidays | **Lead**: meter spikes vs real events |
| **Missing values (L7–8)** | **Lead**: hidden zeros (MNAR) | missing scores | **Lead**: skipped sensitive questions | — | **Lead**: meter gaps & imputation |
| **Text features (L9)** | — | — | **Lead**: Malaysia course fuzzy match | menu item descriptions | — |
| **Shape transformation (L11)** | ✓ | long → wide weekly matrix | ✓ | ✓ | **Lead**: time series → daily-profile matrix |
| **Encoding + PCA/SVD (L12–13)** | ✓ one-hot + SVD | ✓ | ✓ | ✓ | **Lead**: PCA of daily load shapes |
| **t-SNE (L14)** | ✓ | ✓ | ✓ | — | ✓ |
| **Autoencoders (L16)** | anomaly score | — | — | — | **Lead**: anomaly detection on daily profiles |
| **Association rules (L18)** | rules → dropout | clicks → withdrawal | **Lead**: behaviours → sadness | **Lead**: meal attributes × customer group | calendar/occupancy → high use |
| **Clustering (L19–21)** | personas, DBSCAN | engagement trajectories | behaviour profiles | menu/customer clusters | **Lead**: load-shape clusters, DBSCAN |
| **Hypothesis tests** | χ², z, F | z/t: early clicks vs outcome | χ² + survey weighting | **Lead**: base vs intervention (randomised) | t-test: exam vs normal weeks |

---

## 5. Contract for `campuspulse/common/decisions.py`

Every pulse exposes:
```python
DECISIONS = {
    "param_name": ["default_option", "alt_option_1", "alt_option_2"]
}
FINDINGS = {
    "KEY1": "Plain language claim statement"
}
def run_pipeline(decisions: dict) -> dict[str, float]:
    ...
```
A positive finding value indicates that the claim holds. `robustness_matrix` evaluates whether the finding retains its sign and at least 50% of its magnitude under alternative decisions.

