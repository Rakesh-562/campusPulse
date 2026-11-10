# CampusPulse

**Five vital signs of a campus, one shared method.**

CampusPulse analyses five real, public datasets, each a different "vital sign" of a university:
who drops out, who goes quiet online, how students feel, what they eat, and how campus buildings behave.
One method runs through all five. Every data-cleaning decision is logged, and every headline finding is
recomputed under each reasonable alternative choice. That shows which conclusions are solid and which
depend on how the data was cleaned.

> **Umbrella question:** *Can routine campus data (records, clicks, surveys, sales, meters) warn a university
> early enough to act, and how much do those warnings depend on our data-cleaning decisions?*

This is a course project for the Data Analytics course. It covers data processing, statistics and hypothesis
testing, dimensionality reduction, association rules and clustering, plus a dashboard.

---

## Contents
1. [Why this project](#1-why-this-project)
2. [How the project evolved](#2-how-the-project-evolved)
3. [The five vital signs](#3-the-five-vital-signs)
4. [The shared method](#4-the-shared-method)
5. [Results so far](#5-results-so-far)
6. [Quick start](#6-quick-start)
7. [The dashboard](#7-the-dashboard)
8. [Repository layout](#8-repository-layout)
9. [Datasets, sources and licences](#9-datasets-sources-and-licences)
10. [Course syllabus coverage](#10-course-syllabus-coverage)
11. [Adding or changing a pulse](#11-adding-or-changing-a-pulse)
12. [Testing](#12-testing)
13. [Limitations](#13-limitations)
14. [Implications and further directions](#14-implications-and-further-directions)
15. [Presentation guide](#15-presentation-guide)
16. [Open items](#16-open-items)
17. [References](#17-references)

---

## 1. Why this project

Universities already collect a lot of routine data: enrolment and fee records, learning-platform clicks,
wellbeing surveys, canteen tills and electricity meters. Most of it is only used for administration.
If any of it could signal trouble *early*, a university could act before a student fails, withdraws or
burns out.

There's a catch that most dashboards hide. Every number depends on cleaning decisions made along the way:
what a grade of 0 means, whether to smooth a noisy series, what to do with missing survey answers.
Two careful analysts can make different, defensible choices and reach different conclusions. So CampusPulse
asks two questions for every pulse:

1. **What does the data say?** (the finding)
2. **Would the answer change under a different reasonable cleaning choice?** (its robustness)

---

## 2. How the project evolved

| Version | What happened |
|---|---|
| **v1** | Exploratory analysis of three student datasets (a "Student Lifestyle" set, UCI secondary-school performance, a Malaysian mental-health survey). |
| **v2** | **We found the main dataset was generated, not collected.** All five "hours per day" columns add up to exactly 24.0 in every one of its 2,000 rows, physical activity is simply 24 minus the rest, and a depth-3 decision tree reproduces its "stress level" with 100% accuracy (stress was a formula of study and sleep hours). Our earlier findings were just that formula. We switched to real data. |
| **v3** | Five real problems, one per team member, under one theme and one method. |

The evidence is reproduced in [notebooks/00_forensics_why_we_switched.ipynb](notebooks/00_forensics_why_we_switched.ipynb)
and on the dashboard's overview page.

---

## 3. The five vital signs

| # | Pulse | Real problem | Dataset | Main data challenge |
|---|---|---|---|---|
| ① | **Academic** | Who drops out of university, and how early can we tell? | UCI 697 *Predict Students' Dropout and Academic Success*: 4,424 students, Polytechnic Institute of Portalegre (Portugal) | 538 students were enrolled in units but have a semester-1 grade of 0: these zeros are **missing exams, not scores** (92% of them dropped out). 794 students were still "Enrolled" when data collection ended, so their outcome is unknown. Unemployment, inflation and GDP take only 10 values, one per entry cohort. |
| ② | **Engagement** | From which week do learning-platform clicks predict withdrawal? | OULAD (Open University, UK): 32,593 students, 22 course runs, ~10.6M click events | Spiky click logs (smoothing), missing early assessment scores, students who had already left before the cutoff (outcome leakage). |
| ③ | **Wellbeing** | Which behaviours co-occur with persistent sadness? | CDC YRBS 2023 national high-school survey (20,103 students) + Malaysia university mental-health survey (n = 101) | Skipped answers: ~1% skip the sadness question but 13% skip sleep and 24% skip social media. Survey weights. Question numbers change every YRBS year, so `scripts/preprocess_yrbs.py` gives every column a plain name. 49 spellings of course names in the Malaysia survey. |
| ④ | **Food** | Does offering more vegetarian meals cost the canteen sales? | NOVANIMAL (ZHAW, Switzerland): 26,340 sales records from 2 university canteens over a 12-week **randomised** experiment | Large weekday swings in attendance, outlier days, aggregated sales. This is the one pulse that supports cause-and-effect claims. |
| ⑤ | **Energy** | When does a campus building behave differently, and why? | I-BLEND (IIIT Delhi, **an Indian campus**): 52 months of 1-minute smart-meter data and the institute calendar | Hostel meters sent no reading in ~25% of hours. The calendar marks teaching days and low-activity days but **no exam dates**, so we make no exam claims. Delhi's long vacation is also its hottest season, which can fake a "term" effect. |

**How the five connect.** All five pulses follow an academic calendar: semesters, course weeks,
experiment weeks, teaching days and vacations. Two pairs are compared directly:
- **① ↔ ②:** does early data warn as well on a physical campus (Portugal) as at an online university (UK)?
- **③ ↔ ⑤:** students' reported sleep vs how much the campus is awake at night in term vs vacation. This is an *indirect* signal, never a measurement of sleep.

We never join rows across datasets. They describe different people in different countries, so we compare *conclusions*.

---

## 4. The shared method

Every pulse lives in `campuspulse/<pulse>/pipeline.py` and exposes the same contract:

```python
DECISIONS = {"zero_grade": ["nan_plus_flag", "as_score", "drop"], ...}   # first option = our default
FINDINGS  = {"A1": "Debtors drop out more than non-debtors (rate difference)", ...}
def clean(decisions, log=None): ...           # every step logged: rows before/after + reason
def run_pipeline(decisions) -> dict: ...      # {finding_id: signed value}; positive = the claim holds
```

**Decision log.** `campuspulse/common/decisions.py::DecisionLog` records every cleaning step with the row
counts before and after, the option chosen, and why.

**Finding Robustness Matrix.** `robustness_matrix()` reruns the whole pipeline once for every alternative option,
changing one decision at a time, and compares each finding with our default:

| Status | Meaning |
|---|---|
| **holds** | same sign, size between ½× and 2× of the default |
| **weakens** | same sign, but shrinks below ½× |
| **inflates** | same sign, but grows beyond 2×. This is a warning, not good news: the cleaning choice is creating most of the effect |
| **reverses** | the sign flips, so the claim no longer holds |
| **n/a** | the finding cannot be computed under that option (e.g. a group was filtered out) |

`fragility_table()` ranks findings by how far any single choice can move them. A test in
`tests/test_cross_pulse.py` fails if a decision changes no finding at all, so every decision has to matter.

**No hand-typed results.** The conclusions notebook and the dashboard compute every number from the pipelines.
Hypothesis-test conclusions are generated from the p-value by `stats_tests.conclude()`, never written in advance.

---

## 5. Results so far

Computed on the real data for all five pulses (from `scripts/export_dashboard_data.py`, Nov 2026).

### ① Academic: university dropout (3,630 students with a known outcome)
| Finding | Value | Robustness |
|---|---|---|
| A1 Debtors drop out more | **+41.1 percentage points** | holds under every alternative |
| A2 Scholarship holders drop out less | **+34.5 pp** | holds |
| A3 Dropouts have lower semester-1 grades | **+0.98 grade points** (out of 20) | **inflates ×5.5 (to +5.39)** if ungraded zeros are treated as real scores |
| A4 Debt matters *within* semester-1 grade bands | **+23.8 pp** on average: +1.9 pp in the failing band, **+48.1 pp** in the middle band, +21.3 pp among students who passed everything | holds |

**How early can a university warn?** If it contacts the 25% of students with the highest predicted risk
(5-fold cross-validated logistic regression), it reaches:

| Decision point | ROC AUC | Share of eventual dropouts reached | Most possible with a list that size |
|---|---|---|---|
| At enrolment | 0.852 | 54.1% | 63.9% |
| After semester 1 | 0.935 | **62.6%** | 63.9% |
| After semester 2 | 0.949 | 63.4% | 63.9% |

After semester 1 the list is already almost as good as it can get, so **semester 1 is the moment to act**.
Semester 2 adds almost nothing.

### ④ Food: the randomised canteen experiment
| Finding | Value | Robustness |
|---|---|---|
| F1 More vegetarian meals did not cost sales (non-inferiority, 5% margin) | difference **+0.5 meals/day**; the one-sided 95% lower bound (−11.4) is inside the −20.1 margin, so **proven non-inferior** | **reverses** on raw daily totals (lower bound −26.0): the Friday dip hides the result. Removing the weekday effect (Lecture 5) is what makes it provable |
| F2 Vegetarian/vegan share rises in intervention weeks | **+19.3 pp** (≈ 28% → 48%) | **weakens 2.5×** under a 5-day rolling mean, which blends baseline and intervention days |
| F3 Students choose vegetarian slightly more than staff | **+1.0 pp** | holds (n/a when one group is filtered out) |

### ② Engagement: online withdrawal (OULAD, 32,593 student-course rows)
| Finding | Value | Robustness |
|---|---|---|
| E1 Least active learners withdraw more (bottom vs top quarter of early clicks) | **+7.1 pp** | **inflates ×6.4 (to +45.4 pp)** if students who had already left are kept |
| E2 Persisters score higher on the first assessment | **+1.7 points** | inflates when missing scores are set to 0 or dropped; **reverses** if students who had left are kept |
| E3 Withdrawers' activity falls faster | **+0.037** | reverses at a week-2 cutoff, inflates at week 6 |

**Outcome leakage.** 5,343 of the 10,156 withdrawals happened before week 5. Their low clicks are a *result* of
leaving, not a warning. Keeping them makes early clicks look like a strong predictor (ROC AUC 0.73); removing them
leaves ROC AUC ≈ 0.56 at weeks 2, 4 and 6. On its own, early platform activity is a weak warning sign.

### ③ Wellbeing: CDC YRBS 2023 (16,900 students who answered every behaviour question)
| Finding | Value | Robustness |
|---|---|---|
| W1 Students sleeping under 8 hours report persistent sadness more often | **+18.1 pp** (weighted) | holds |
| W2 Bullied students (school or online) report it more often | **+34.3 pp** | holds |
| W3 Inactive students (0 active days a week) report it more often | **+10.6 pp** | holds |
| W4 Malaysia survey: course is associated with depression | Cramér's V **0.25** (p = 0.41, not significant) | **inflates to 0.71** on raw course text: 48 tiny "courses" create a fake association |

Persistent sadness falls from 60% at ≤4 hours of sleep to 26% at 8 hours, then rises again at 10+ hours (34%).

### ⑤ Energy: IIIT Delhi (38,520 hours, Aug 2013 – Dec 2017)
| Finding | Value | Robustness |
|---|---|---|
| EN1 Academic building uses more power on teaching days (9:00–18:00) | **+17.9 kW** (Cohen's d 1.32) | holds |
| EN2 Hostels use more power at night in term than in vacation | **+12.2 kW** (d 0.97) | holds |
| EN3 The same comparison within the same calendar month | **+10.4 kW** | weakens with forward-filled gaps; only 7 months contain both term and vacation (p = 0.06) |

### The cleaning choices that would have misled a university
1. **Engagement E1:** keep students who had already left, and early clicks look 6× more predictive than they are.
2. **Food F1:** use raw daily totals, and a real "no loss in sales" result becomes "not proven".
3. **Academic A3:** treat ungraded students' 0 as a real exam score, and the grade gap looks 5.5× bigger than it is.
4. **Wellbeing W4:** skip the course-name cleaning, and a non-existent course–depression association triples.

---

## 6. Quick start

Requires Python 3.10+.

```powershell
# Windows PowerShell (macOS/Linux: python3 -m venv .venv; source .venv/bin/activate)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

pytest                                   # 11 tests; data-dependent ones skip if data is missing
python scripts/export_dashboard_data.py  # compute everything the dashboard shows
start dashboard\index.html               # open the dashboard (macOS: open, Linux: xdg-open)
```

**Get the data for the other pulses.** UCI dropout, NOVANIMAL and the Malaysia survey are already in `data/raw/`.
The other three are downloaded, unpacked and turned into the tables the pipelines read:

```powershell
python scripts/download_data.py --pulse all   # OULAD (47 MB zip), I-BLEND (590 MB zip), CDC YRBS 2023 (8.5 MB)
python scripts/preprocess_oulad.py            # -> data/processed/engagement/weekly_engagement.csv
python scripts/preprocess_yrbs.py             # -> data/processed/wellbeing/yrbs_2023.csv
python scripts/preprocess_iblend.py           # -> data/processed/energy/hourly_campus_power.csv
```

**Rebuild the notebooks** (executed in-process, outputs embedded):

```powershell
python scripts/generate_all_executed_notebooks.py   # five pulse notebooks + forensics + cross-pulse
python scripts/generate_conclusions_nb.py           # notebooks/conclusions.ipynb
```

Run these after the data steps above. Otherwise the notebooks for missing pulses fill with errors.

---

## 7. The dashboard

`dashboard/` is a plain HTML/CSS/JavaScript site with no framework, no build step and no server.
`dashboard/data.js` is generated by `scripts/export_dashboard_data.py`. The page only reads and displays it,
so it can never disagree with the pipelines.

| Page | What it shows |
|---|---|
| **Overview** | The umbrella question, one card per pulse, and the cleaning choices that change the answer |
| **① to ⑤ pulse pages** | A row of buttons, one per deliverable; only the chosen one is shown, so the page stays short |
| **Robustness** | Every robustness matrix (one collapsible card per pulse) |
| **Method** | The pipeline, why we changed dataset, limitations |

The deliverable buttons on every pulse page:

| Button | What it shows |
|---|---|
| **Findings** | The headline findings, a "try a different cleaning choice" selector, and the pulse's key chart |
| **Data processing** | The pulse's main data challenge (hidden zeros, leakage, skipped answers, weekday noise, meter gaps) and the cleaning log |
| **Statistics** | Descriptive statistics, distributions and hypothesis tests with effect sizes |
| **Dimensionality reduction** | PCA and t-SNE maps coloured by outcome, variance explained, top loadings |
| **Association rules** | Apriori rules with support, confidence and lift |
| **Clustering** | K-Means groups on a PCA map, four methods compared, what each group looks like |

The URL remembers the selection (for example `index.html#/academic/statistics`), so a slide can link straight to one view.
The code is plain JavaScript: `app.js` has one function per page and per panel, and `charts.js` one function per chart type.

Charts are SVG with hover tooltips and a "Show as table" view. Colours come from a palette checked for
colour-blind separation and contrast, and status is always shown as a dot plus a word, never by colour alone.
The page follows the system light/dark setting and has a toggle. Pulses without local data say so.

Open it by double-clicking `dashboard/index.html`, or serve it with `python -m http.server -d dashboard 8000`
and visit http://localhost:8000.

---

## 8. Repository layout

```
campusPulse/
├── README.md                 ← this file
├── PLAN.md                   ← project plan (five pulses, roles)
├── requirements.txt
├── campuspulse/              ← the analysis package
│   ├── common/
│   │   ├── decisions.py      ← DecisionLog, robustness_matrix, status_matrix, fragility_table
│   │   ├── stats_tests.py    ← z, Welch t, ANOVA F, χ², effect sizes, Bonferroni, conclude()
│   │   ├── reduce.py         ← PCA, TruncatedSVD, t-SNE, autoencoder anomaly score
│   │   ├── clustering.py     ← K-Means, Ward, DBSCAN (k-distance eps), OPTICS; scaling + random sampling
│   │   ├── rules.py          ← Apriori (support / confidence / lift), target-only consequents
│   │   └── plots.py          ← shared matplotlib theme
│   ├── academic/pipeline.py  ← ① incl. early-warning timeline, stratified debt effect
│   ├── engagement/pipeline.py← ② incl. leakage guard, week-by-week timeline
│   ├── wellbeing/pipeline.py ← ③ incl. survey weights, course-text normalisation
│   ├── food/pipeline.py      ← ④ incl. weekday adjustment, non-inferiority test
│   └── energy/pipeline.py    ← ⑤ incl. same-month term-vs-vacation check (season confound)
├── dashboard/                ← plain HTML/CSS/JS dashboard (index.html, app.js, charts.js, styles.css, data.js)
├── notebooks/
│   ├── 00_forensics_why_we_switched.ipynb
│   ├── academic/01_academic_pulse.ipynb … energy/05_energy_pulse.ipynb
│   ├── 99_cross_pulse_links.ipynb
│   └── conclusions.ipynb     ← every number computed, none typed
├── scripts/
│   ├── download_data.py
│   ├── preprocess_oulad.py / preprocess_yrbs.py / preprocess_iblend.py   ← raw → processed tables
│   ├── export_dashboard_data.py
│   ├── generate_all_executed_notebooks.py
│   ├── generate_conclusions_nb.py
│   └── notebook_builder.py
├── data/
│   ├── README.md             ← data cards for every dataset
│   ├── raw/                  ← uci_dropout/, novanimal/, student_mh/, legacy/ (committed); oulad/, yrbs/, iblend/ (git-ignored)
│   └── processed/            ← git-ignored
└── tests/                    ← pipeline, contract and robustness tests
```

---

## 9. Datasets, sources and licences

| Pulse | Dataset | Source | Licence |
|---|---|---|---|
| ① | Predict Students' Dropout and Academic Success (UCI 697) | https://archive.ics.uci.edu/dataset/697 | CC BY 4.0 |
| ② | Open University Learning Analytics Dataset (OULAD) | https://analyse.kmi.open.ac.uk/open-dataset (UCI mirror DOI 10.24432/C5KK69) | CC BY 4.0 |
| ③ | Youth Risk Behavior Surveillance System (YRBS) | https://www.cdc.gov/yrbs/ | US public domain |
| ③ | Student Mental Health survey (Malaysia, 2020) | Kaggle | open dataset |
| ④ | NOVANIMAL aggregated menu sales | https://zenodo.org/records/3890931 | CC BY 4.0 |
| ⑤ | I-BLEND campus electricity | https://doi.org/10.6084/m9.figshare.c.3893581 | CC BY 4.0 |
| legacy | Student Lifestyle dataset (synthetic) | Kaggle / OpenML | used only to show why we switched |

Full data cards (size, time span, quirks, evidence that each dataset is real) are in [data/README.md](data/README.md).

---

## 10. Course syllabus coverage

| Topic | Where it is applied |
|---|---|
| Collecting & storing (L2) | Large click logs and 1-minute meter data → weekly and hourly tables; header and encoding fixes |
| One- and two-variable statistics (L3–4) | Dropout rates by group, sleep vs sadness, daily sales, teaching vs low-activity days |
| Noise removal / smoothing (L5) | **Food:** weekday adjustment vs rolling mean · Engagement: rolling and exponential smoothing · Energy: rolling median |
| Outliers (L6–7) | **Food:** IQR outlier days kept / dropped / winsorized · Energy: 99th-percentile clipping · Academic: IQR on age |
| Missing values (L7–8) | **Academic:** ungraded zeros as missing-not-at-random · **Wellbeing:** skipped sensitive answers · **Energy:** meter gaps (interpolate / forward-fill / drop) |
| Text features (L9) | **Wellbeing:** 49 course spellings → 7 disciplines · Food: menu keyword flags |
| Graph features (L10) | Planned (student-similarity graph); not part of the current results |
| Shape transformation (L11) | **Energy:** series → day × 24-hour matrix · Engagement: click log → weekly matrix · Food: sales → day × meal type |
| Encoding, PCA, SVD (L12–13) | One-hot encoding of coded categories, TruncatedSVD on the one-hot matrix, PCA on quantities |
| t-SNE (L14) | 2-D embeddings for academic, engagement, wellbeing and energy |
| Autoencoders (L16) | Reconstruction error as an anomaly score (academic students, energy daily profiles) |
| Association rules (L18) | Apriori with dropout, withdrawal, sadness, vegetarian choice and high demand as targets |
| Clustering (L19–21) | K-Means, Ward, DBSCAN with k-distance eps, OPTICS on standardised features; clusters named and outcome rates reported; DBSCAN noise profiled as "fits no group" |
| Hypothesis testing | χ², two-proportion z, Welch t, ANOVA F, stratified χ², non-inferiority t, same-month one-sample t, Bonferroni, effect sizes |

---

## 11. Adding or changing a pulse

1. Copy the contract from `campuspulse/academic/pipeline.py`: `DECISIONS`, `FINDINGS`, `clean()`, `run_pipeline()`.
2. Make every finding a signed number where **positive means the claim holds**.
3. Make every decision matter: the cross-pulse test fails if an option changes no finding.
4. Put anything two pulses need in `campuspulse/common/`; never copy code between pulses.
5. Subsample with `clustering.sample_rows()` (random), never `df.iloc[:n]`. Most tables are sorted.
6. Add the pulse to `META` and an `extras_<pulse>()` function in `scripts/export_dashboard_data.py`, then a `body<Pulse>()` renderer in `dashboard/app.js`.
7. Run `pytest`, then `python scripts/export_dashboard_data.py`.

---

## 12. Testing

```powershell
pytest
```

- `test_decisions.py`: status categories (holds / weakens / inflates / reverses / n/a), fragility ranking.
- `test_cross_pulse.py`: every pulse meets the contract, and **every decision changes at least one finding**.
- `test_<pulse>.py`: each pipeline runs on its real data and returns finite values (skipped if the data is absent).
  The tests check the code, not the conclusions: a finding is allowed to come out negative.

---

## 13. Limitations

- **Not our university.** Five institutions in four countries. The method transfers; the numbers may not.
- **Association, not causation**, everywhere except the randomised canteen experiment (④).
- **YRBS is a high-school survey.** The Malaysia survey has only 101 respondents, so W4 is exploratory.
- **Hostel electricity is an indirect signal** of activity at night. It does not measure anyone's sleep, and the I-BLEND calendar has no exam dates.
- **One decision at a time.** The robustness matrix does not test combinations of decisions.
- **Outcome censoring.** 18% of UCI students were still enrolled. We test excluding them vs counting them as non-dropouts; neither is perfect.
- **Cohort-level economics.** Unemployment, inflation and GDP are identical within an entry year, so they are treated as cohort labels, not student traits (avoiding the ecological fallacy).

## 14. Implications and further directions

**Implications (where the computed results support them)**
- A semester-1 checkpoint can reach most future dropouts with a short contact list. Fee status and scholarships are cheap, actionable signals, and A4 shows *where* debt matters most (the middle grade band, not students already failing).
- Canteens can expand vegetarian and vegan options without losing sales. This comes from a randomised experiment.
- Published campus statistics should state their cleaning decisions: one reasonable choice can create, inflate or hide most of an effect.

**Further directions**
- Indian institutional data (AISHE, or our own university with ethics approval) to test which signals transfer.
- Test combinations of decisions (full factorial), not just one at a time.
- Design-based survey tests for YRBS (Rao–Scott χ²) instead of weighted rates only.
- Graph features (Lecture 10): a student-similarity graph and community detection compared with the clusters.

---

## 15. Presentation guide

7 minutes plus 3 minutes of Q&A. The five evaluation criteria are equally weighted.

| Time | Content | Criterion |
|---|---|---|
| 0:00–1:00 | Why campus data matters + "our first dataset was fake" (24-hour sums, 100% tree) | Motivation |
| 1:00–4:00 | Dashboard demo: one pulse each, one insight + one data challenge; flip a decision live | Clarity of work |
| 4:00–5:30 | The cleaning choices that would have misled a university (leakage, weekday noise, hidden zeros); the semester-1 timeline; the same-month energy check | Novelty |
| 5:30–7:00 | Limitations, implications, next steps | Limitations & implications |

Likely questions: Why is a 0 grade missing and not a low score? How did you treat "Enrolled"? Is debt a cause
or a symptom? Why remove students who had already left (OULAD)? Why compare within the same month for energy?
Why no exam analysis (the calendar has no exam dates)? How do you know the first dataset was synthetic?
Does a Portuguese polytechnic tell us anything about India? Why non-inferiority instead of a plain t-test?

---

## 16. Open items

- **Commit choice:** raw OULAD, YRBS and I-BLEND files and `data/processed/` are git-ignored (large). Anyone can rebuild them with the download and preprocess commands above.
- **Replaced teammate files:** the earlier processed files for ②, ③ and ⑤ were made with an unknown script. The new preprocessing found two problems in them: YRBS question numbers that don't match the 2023 file (sleep is Q85, not Q88), and an `is_exam` column that the I-BLEND calendar can't provide. Everything is now rebuilt from the official raw files.
- **Graph features (L10)** are planned, not implemented.

---

## 17. References

- Realinho, V., Machado, J., Baptista, L., & Martins, M. V. (2022). Predicting student dropout and academic success. *Data*, 7(11), 146.
- Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). Open University Learning Analytics dataset. *Scientific Data*, 4, 170171.
- Centers for Disease Control and Prevention. Youth Risk Behavior Surveillance System (YRBSS). https://www.cdc.gov/yrbs/
- Egeler, C., & Baur, I. (2020). NOVANIMAL aggregated menu sales data. Zenodo. https://doi.org/10.5281/zenodo.3890931
- Rashid, H., Singh, P., & Singh, A. (2019). I-BLEND, a campus-scale commercial and residential buildings electrical energy dataset. *Scientific Data*, 6, 190015.
