# CampusPulse

**Three vital signs of a campus, one shared method.**

CampusPulse analyses three real, public datasets, each a different "vital sign" of a university: who drops out,
how students feel, and how campus buildings behave. One method runs through all three. Every data-cleaning decision
is logged, and every headline finding is recomputed under each reasonable alternative choice. That shows which
conclusions are solid and which depend on how the data was cleaned.

> **Umbrella question:** *Can routine campus data (records, surveys, meters) warn a university early enough to act,
> and how much do those warnings depend on our data-cleaning decisions?*

A course project for the Data Analytics course: data processing, statistics and hypothesis testing, dimensionality
reduction, association rules and clustering, plus a dashboard. **How to use the dashboard: [DASHBOARD_GUIDE.md](DASHBOARD_GUIDE.md).**

---

## Contents
1. [Why this project](#1-why-this-project)
2. [How the project evolved](#2-how-the-project-evolved)
3. [The three vital signs](#3-the-three-vital-signs)
4. [The shared method](#4-the-shared-method)
5. [Results](#5-results)
6. [Quick start](#6-quick-start)
7. [The dashboard](#7-the-dashboard)
8. [Repository layout](#8-repository-layout)
9. [Datasets, sources and licences](#9-datasets-sources-and-licences)
10. [Course syllabus coverage](#10-course-syllabus-coverage)
11. [Testing](#11-testing)
12. [Limitations](#12-limitations)
13. [Implications and further directions](#13-implications-and-further-directions)
14. [References](#14-references)

---

## 1. Why this project

Universities already collect routine data: enrolment and fee records, wellbeing surveys, electricity meters.
If any of it could signal trouble *early*, a university could act before a student fails or burns out.

The catch most dashboards hide: every number depends on cleaning decisions made along the way, such as what a
grade of 0 means or what to do with skipped survey answers. Two careful analysts can make different, defensible
choices and reach different conclusions. So CampusPulse asks two questions for every finding:

1. **What does the data say?** (the finding, with a graph and a simple hypothesis test)
2. **Would the answer change under a different reasonable cleaning choice?** (its robustness)

---

## 2. How the project evolved

| Version | What happened |
|---|---|
| v1 | Exploratory analysis of a "Student Lifestyle" dataset and two smaller student datasets |
| v2 | **We found the main dataset was generated, not collected.** The five "hours per day" columns add up to exactly 24.0 in all 2,000 rows, and a depth-3 decision tree reproduces its "stress level" with 100% accuracy. Our earlier findings were just that formula, so we switched to real data |
| v3 | Five real problems, one per team member |
| v3.1 (current) | Narrowed to the three pulses we present (Academic, Wellbeing, Energy); each finding now has its own graph and a simple z, t or chi-square test |

The evidence is in [notebooks/00_forensics_why_we_switched.ipynb](notebooks/00_forensics_why_we_switched.ipynb)
and on the dashboard's Method page.

---

## 3. The three vital signs

| # | Pulse | Question | Dataset | Main data challenge |
|---|---|---|---|---|
| ① | **Academic** | Who drops out of university, and how early can we tell? | UCI 697 *Predict Students' Dropout and Academic Success*: 4,424 students, Polytechnic Institute of Portalegre (Portugal) | 538 students were enrolled in units but graded 0: **missing exams, not scores** (91.6% of them dropped out). 794 students were still "Enrolled" when data collection ended. Unemployment, inflation and GDP take only 10 values, one per entry cohort |
| ② | **Wellbeing** | Which behaviours co-occur with persistent sadness? | CDC YRBS 2023 national high-school survey (20,103 students) + Malaysia university mental-health survey (n = 101) | ~1% skip the sadness question but 13% skip sleep and 24% skip social media. Survey weights. YRBS question numbers change every year, so `scripts/preprocess_yrbs.py` names every column. 49 spellings of course names in the Malaysia survey |
| ③ | **Energy** | When does a campus building behave differently, and why? | I-BLEND (IIIT Delhi, an Indian campus): 52 months of 1-minute smart-meter data + institute calendar | Hostel meters sent no reading in ~25% of hours. The calendar marks teaching and low-activity days but **no exam dates**. Delhi's long vacation is also its hottest season |

We never join rows across datasets: they describe different people in different countries, so we compare *conclusions*.

---

## 4. The shared method

Every pulse lives in `campuspulse/<pulse>/pipeline.py` and exposes the same contract:

```python
DECISIONS = {"zero_grade": ["nan_plus_flag", "as_score", "drop"], ...}   # first option = our default
FINDINGS  = {"A1": "Debtors drop out more than non-debtors (rate difference)", ...}
def clean(decisions, log=None): ...           # every step logged: rows before/after + reason
def run_pipeline(decisions) -> dict: ...      # {finding_id: signed value}; positive = the claim holds
def finding_evidence() -> dict: ...           # per finding: the bars of its graph + one simple hypothesis test
```

**Decision log.** `DecisionLog` records every cleaning step with the rows before and after, the option chosen, and why.

**Simple hypothesis tests.** Each finding is tested with the simplest course test for its comparison:
two-proportion **z-test** (two percentages), two-sample **t-test** (two means), **chi-square** test (two categories),
one-sample **t-test** (a set of differences against 0). `stats_tests.conclude()` and `test_result()` write the decision
from the p-value; nothing is typed in advance.

**Finding Robustness Matrix.** `robustness_matrix()` reruns the pipeline once for every alternative option, one
decision at a time, and labels each finding: **holds** (same sign, ½× to 2× of the default), **weakens** (below ½×),
**inflates** (above 2×: the cleaning choice creates most of the effect), **reverses** (sign flips), **n/a**.
A test fails if a decision changes no finding at all, so every decision has to matter.

---

## 5. Results

Computed on the real data (`scripts/export_dashboard_data.py`, Nov 2026).

### ① Academic: university dropout (3,630 students with a known outcome)
| Finding | Value | Simple test | Robustness |
|---|---|---|---|
| A1 Debtors drop out more | **+41.1 pp** (75.5% vs 34.5%) | z = 16.10, p < 0.0001 | holds |
| A2 Scholarship holders drop out less | **+34.5 pp** (13.8% vs 48.4%) | z = 18.86, p < 0.0001 | holds |
| A3 Dropouts have lower semester-1 grades | **+0.98 points** (13.1 vs 12.1 of 20) | t = 18.41, p < 0.0001 | **inflates ×5.5** if ungraded zeros are treated as scores |
| A4 Debt matters within semester-1 grade bands | **+23.8 pp** on average: +1.9 / +48.1 / +21.3 pp | z per band: failing band p = 0.11 (not significant); other two p < 0.0001 | holds |

**How early can a university warn?** Contacting the 25% highest-risk students reaches 54.1% of dropouts at enrolment,
**62.6% after semester 1** and 63.4% after semester 2, against a 63.9% maximum. Semester 1 is the moment to act.

### ② Wellbeing: CDC YRBS 2023 (16,900 students who answered every behaviour question)
| Finding | Value | Simple test | Robustness |
|---|---|---|---|
| W1 Sleep under 8 hours → more persistent sadness | **+18.1 pp** (44.5% vs 26.4%) | z = 20.50, p < 0.0001 | holds |
| W2 Bullied → more sadness | **+34.3 pp** (66.1% vs 31.8%) | z = 38.54, p < 0.0001 | holds |
| W3 Inactive (0 days) → more sadness | **+10.6 pp** (49.3% vs 38.7%) | z = 9.01, p < 0.0001 | holds |
| W4 Malaysia: course and depression | no link | chi-square = 6.08, p = 0.41 | **inflates** on raw course text (48 tiny "courses" fake a link) |

Sadness falls from 60% at ≤4 hours of sleep to 26% at 8 hours, then rises to 34% at 10+ hours.

### ③ Energy: IIIT Delhi (38,520 hours, Aug 2013 – Dec 2017)
| Finding | Value | Simple test | Robustness |
|---|---|---|---|
| EN1 Academic building uses more on teaching days (9:00–18:00) | **+17.9 kW** (45.5 vs 27.7) | t = 63.21, p < 0.0001 | holds |
| EN2 Hostels use more at night in term than vacation | **+12.2 kW** (54.5 vs 42.3) | t = 31.63, p < 0.0001 | holds |
| EN3 Same comparison within the same month | **+10.4 kW** | one-sample t = 2.30, p = 0.061 (not significant, 7 months) | weakens with forward-filled gaps |

### The cleaning choices that would have changed the answer
1. **Academic A3:** treat ungraded students' 0 as a real score, and the grade gap looks 5.5× bigger than it is.
2. **Wellbeing W4:** skip the course-name cleaning, and a non-existent course–depression link appears.
3. **Energy EN3:** fill meter gaps by repeating the last reading, and the same-month effect halves.

---

## 6. Quick start

Requires Python 3.10+.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1                  # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python scripts/download_data.py --pulse all   # I-BLEND (590 MB zip) and CDC YRBS 2023; UCI 697 is in the repo
python scripts/preprocess_yrbs.py             # -> data/processed/wellbeing/yrbs_2023.csv
python scripts/preprocess_iblend.py           # -> data/processed/energy/hourly_campus_power.csv

pytest                                        # 9 tests
python scripts/export_dashboard_data.py       # compute everything the dashboard shows
start dashboard\index.html                    # open it (macOS: open, Linux: xdg-open)
```

Rebuild the executed notebooks after the data steps:

```powershell
python scripts/generate_all_executed_notebooks.py   # academic, wellbeing, energy, forensics, cross-pulse
python scripts/generate_conclusions_nb.py           # notebooks/conclusions.ipynb
```

---

## 7. The dashboard

`dashboard/` is plain HTML/CSS/JavaScript: no framework, no build step, no server. It only displays
`dashboard/data.js`, which `scripts/export_dashboard_data.py` writes, so it can never disagree with the pipelines.

Each pulse page has six numbered tabs, in presentation order:

| Tab | What it shows |
|---|---|
| 1. Data processing | The main data problem, its graphs, and the cleaning log |
| 2. Statistics | Descriptive statistics, distributions, one simple test per finding |
| 3. Dimensionality reduction | PCA and t-SNE maps coloured by outcome, variance explained, loadings |
| 4. Association rules | Two Apriori rules per pulse that sound like common sense but have lift below 1 |
| 5. Clustering | K-Means groups on a PCA map, four methods compared, group profiles |
| 6. Findings | Every finding with its own graph and test, the chart that answers the question, the cleaning-choice selector |

The URL keeps the view (e.g. `index.html#/academic/findings`). `app.js` has one plain function per page and per
tab; `charts.js` one function per chart type. Full walkthrough: [DASHBOARD_GUIDE.md](DASHBOARD_GUIDE.md).

---

## 8. Repository layout

```
campusPulse/
├── README.md, DASHBOARD_GUIDE.md, PLAN.md, requirements.txt
├── campuspulse/
│   ├── common/        decisions.py (log, robustness), stats_tests.py (z, t, chi-square, F),
│   │                  reduce.py (PCA, SVD, t-SNE, autoencoder), clustering.py, rules.py, plots.py
│   ├── academic/pipeline.py    ① incl. early-warning timeline, debt within grade bands
│   ├── wellbeing/pipeline.py   ② incl. survey weights, course-text cleaning
│   └── energy/pipeline.py      ③ incl. same-month term-vs-vacation check
├── dashboard/         index.html, app.js, charts.js, styles.css, data.js (generated)
├── notebooks/         00_forensics, academic/, wellbeing/, energy/, 99_cross_pulse_links, conclusions
├── scripts/           download_data.py, preprocess_yrbs.py, preprocess_iblend.py, export_dashboard_data.py,
│                      generate_all_executed_notebooks.py, generate_conclusions_nb.py, notebook_builder.py
├── data/              raw/ (uci_dropout, student_mh, legacy committed; yrbs, iblend git-ignored), processed/ (git-ignored)
└── tests/
```

---

## 9. Datasets, sources and licences

| Pulse | Dataset | Source | Licence |
|---|---|---|---|
| ① | Predict Students' Dropout and Academic Success (UCI 697) | https://archive.ics.uci.edu/dataset/697 | CC BY 4.0 |
| ② | Youth Risk Behavior Survey 2023, national high school | https://www.cdc.gov/yrbs/data/index.html | US public domain |
| ② | Student Mental Health survey (Malaysia, 2020) | Kaggle | open dataset |
| ③ | I-BLEND campus electricity + calendar | https://doi.org/10.6084/m9.figshare.c.3893581 | CC BY 4.0 |
| legacy | Student Lifestyle dataset (synthetic) | Kaggle / OpenML | used only to show why we switched |

---

## 10. Course syllabus coverage

| Topic | Where it is applied |
|---|---|
| Collecting & storing (L2) | 1-minute meter data → hourly table; CDC fixed-width file read with its SAS layout |
| One- and two-variable statistics (L3–4) | Dropout rates by group, sadness by sleep, power on teaching vs low-activity days |
| Noise removal / smoothing (L5) | Energy: rolling median for spikes |
| Outliers (L6–7) | Energy: 99th-percentile clipping vs raw; Academic: IQR on age |
| Missing values (L7–8) | **Academic:** ungraded zeros as missing-not-at-random · **Wellbeing:** skipped answers (drop / mode / flag) · **Energy:** meter gaps (drop / interpolate / forward-fill) |
| Text features (L9) | **Wellbeing:** 49 course spellings → 7 disciplines |
| Shape transformation (L11) | **Energy:** hourly series → day × 24-hour matrix; Academic: one-hot of coded categories |
| Encoding, PCA, SVD (L12–13) | PCA on quantities, TruncatedSVD on the one-hot matrix |
| t-SNE (L14) | 2-D maps for all three pulses |
| Autoencoders (L16) | Reconstruction error as an anomaly score (academic students, energy days) |
| Association rules (L18) | Apriori; per pulse, two common beliefs the data contradicts (lift < 1), via `rules.surprising_rules` |
| Clustering (L19–21) | K-Means, Ward, DBSCAN (k-distance eps), OPTICS on standardised features |
| Hypothesis testing | Two-proportion z, two-sample t, chi-square, one-sample t (one per finding) |

---

## 11. Testing

`pytest` runs 9 tests: robustness statuses and ranking; every pulse meets the contract; every decision changes at
least one finding; every finding has a graph and a complete test; each pipeline runs on its real data.

---

## 12. Limitations

- **Not our university.** Three institutions in three countries; the method transfers, the numbers may not.
- **Association, not causation** everywhere.
- **YRBS is a high-school survey.** The Malaysia survey has 101 respondents, so W4 is exploratory.
- **No exam dates** in the I-BLEND calendar; hostel electricity is not a measurement of sleep.
- **One decision at a time.** Combinations of cleaning choices are not tested.
- **Outcome censoring.** 18% of UCI students were still enrolled; we exclude them by default and test counting them as non-dropouts.

## 13. Implications and further directions

- A semester-1 checkpoint reaches most future dropouts with a short contact list; fee status and scholarships are cheap signals, and debt matters most in the middle grade band.
- Published campus statistics should state their cleaning decisions: one reasonable choice can create or hide most of an effect.
- Next: Indian institutional data (AISHE, or our own university with ethics approval); test combinations of decisions.

---

## 14. References

- Realinho, V., Machado, J., Baptista, L., & Martins, M. V. (2022). Predicting student dropout and academic success. *Data*, 7(11), 146.
- Centers for Disease Control and Prevention. Youth Risk Behavior Surveillance System (YRBSS). https://www.cdc.gov/yrbs/
- Rashid, H., Singh, P., & Singh, A. (2019). I-BLEND, a campus-scale commercial and residential buildings electrical energy dataset. *Scientific Data*, 6, 190015.
