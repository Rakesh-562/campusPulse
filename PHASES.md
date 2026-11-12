# CampusPulse — Phase-by-Phase Execution Plan

> Companion to [PLAN.md](PLAN.md) (the *what* and *why*). This file is the *how* and *when*.
> Day numbers assume a ~16-day window. If our deadline is different, scale each phase proportionally but **keep the gates**.

**Owners:** P1 = ① Academic (dropout) · P2 = ② Engagement (OULAD) · P3 = ③ Wellbeing (YRBS + Malaysia) · P4 = ④ Food (NOVANIMAL) · P5 = ⑤ Energy (I-BLEND)
**Infra lead:** P1 (repo, `common/`, dashboard shell), because P1's data is already in the repo.

---

## Overview

| Phase | Name | Days | Main output | Gate (must pass to move on) |
|---|---|---|---|---|
| 0 | Setup & shared contracts | 1–2 | Repo skeleton, `common/` interfaces, dashboard shell | Every member can run `streamlit run app/Home.py` locally |
| 1 | Data acquisition & data cards | 1–3 | Raw data downloaded, data card per pulse, first profile | All 5 data cards reviewed; no dataset is synthetic or unusable |
| 2 | Cleaning & signature challenge | 3–6 | Clean data, cleaning log, ≥2 options compared for the signature challenge | **Mid-point sync**: each shows "what's messy and what we did" |
| 3 | Statistics & hypothesis testing | 5–8 | Descriptives, 2-variable relationships, ≥3 hypothesis tests per pulse | Effect sizes + correction on every test |
| 4 | Dimensionality reduction | 7–9 | PCA/SVD + t-SNE or autoencoder per pulse | Each plot has a one-line interpretation |
| 5 | Association rules & clustering | 8–11 | Filtered rules + ≥2 clustering methods, named groups | Rules and clusters interpreted, not just listed |
| 6 | Robustness & cross-pulse links | 10–12 | Robustness Matrix per pulse, 3+ cross-pulse links | Matrices render from the shared helper |
| 7 | Dashboard integration | 9–13 | 5 pulse pages + Calendar + Fragility pages | **Feature freeze, end of Day 12** |
| 8 | Story, slides & Q&A | 13–15 | Slides, script, Q&A bank, backup recording | Two timed run-throughs ≤ 7:00 |
| 9 | Final freeze & submission | 15–16 | Tagged release, README, report | Fresh-clone test passes |

Phases overlap on purpose: dashboard work (Phase 7) starts as soon as each pulse has clean data.

---

## Phase 0 — Setup & shared contracts (Days 1–2)

**Goal:** everyone builds on the same base, so integration in Phase 7 is easy.

### Shared tasks (P1 leads, everyone reviews)
- [ ] Create the folder structure from PLAN.md §5 (`campuspulse/common/`, `campuspulse/<pulse>/`, `notebooks/<pulse>/`, `app/pages/`, `data/raw/<pulse>/`).
- [ ] Move `data/raw/student_lifestyle_dataset.csv`, `student-mat.csv`, `student-por.csv` → `data/raw/legacy/`; Malaysia survey → `data/raw/student_mh/`. Update paths in `src/ingest.py`, then move it to `campuspulse/academic/ingest.py` (keep the legacy loaders in `campuspulse/legacy.py`).
- [ ] Move the old notebooks to `notebooks/legacy/` (we only keep them for the forensics story).
- [ ] `requirements.txt` with pinned versions: `pandas numpy scipy scikit-learn statsmodels matplotlib seaborn plotly streamlit mlxtend networkx rapidfuzz pyarrow pyreadstat`.
- [ ] `.gitignore`: `__pycache__/`, `.ipynb_checkpoints/`, `venv/`, `data/raw/oulad/`, `data/raw/yrbs/`, `data/raw/iblend/`, `data/processed/**/*.parquet`.
- [ ] Install `nbstripout` so committed notebooks have no outputs.
- [ ] Branch rules: `main` protected; each person works on `pulse/<name>`; shared code on `infra/<topic>`.

### Shared contracts (agree in a 30-min call on Day 1, then freeze)
- [ ] **`common/decisions.py`**
  ```python
  class DecisionLog:            # records step, rows_before, rows_after, reason
      def log(self, step, before, after, reason): ...
      def to_frame(self) -> pd.DataFrame: ...

  def robustness_matrix(run_pipeline, decisions: dict, findings: dict,
                        baseline: dict | None = None) -> pd.DataFrame:
      """Runs run_pipeline once per option of each decision (one at a time),
      compares each finding to the baseline: 'holds' | 'weakens' | 'reverses'."""
  ```
- [ ] **Each pulse module exposes:** `load_raw()`, `clean(decisions) -> (df, DecisionLog)`, `DECISIONS`, `FINDINGS`, `run_pipeline(decisions) -> dict[finding_id, float]`, `DATA_CARD` (dict).
- [ ] **`common/stats_tests.py`**: `z_two_means`, `t_two_means`, `f_two_vars`, `chi2_independence` (with expected-count check), each returning statistic, p, effect size (Cohen's d / Cramér's V), and a `bonferroni(pvals)` helper. Reuse the textbook functions from `notebooks/legacy/06_hypothesis_testing.ipynb`.
- [ ] **`common/plots.py`**: one colour palette and a `real_badge(fig, dataset, is_real=True)` helper.
- [ ] **Dashboard shell:** `app/Home.py` with 5 placeholder "vital-sign" cards + empty pages that import from `campuspulse.<pulse>`.

### Deliverables
- Repo skeleton merged into `main`; `common/` stubs with docstrings; dashboard shell runs.

### Gate
- Every member clones, installs, and sees the shell dashboard on their own machine.

---

## Phase 1 — Data acquisition & data cards (Days 1–3)

**Goal:** real data in hand, its quirks known, and nothing surprising later.

### Shared tasks
- [ ] `scripts/download_data.py --pulse <name>`: downloads into `data/raw/<pulse>/` (new empty folder), verifies the file list, and prints a manual link if login is needed.
- [ ] `data/README.md`: one **data card** per pulse with source URL, citation, licence, collection period, population, rows × columns, file sizes, known caveats, **evidence that the data is real**.
- [ ] Each person runs a **first-profile script** (shape, dtypes, missing counts *including disguised ones like 0, -1, 9, blank, "NA"*, duplicates, value ranges, top categories) and saves the output to `reports/<pulse>/profile.md`.

### Per-pulse tasks
| Pulse | Acquisition | Things to find out on Day 1–3 |
|---|---|---|
| ① P1 | ✅ already in `data/raw/uci_dropout/` | Build `data/reference/dropout_codebook.csv` (code → label) from the UCI variables page |
| ② P2 | Download the OULAD zip from the official OU page (or the UCI mirror). Files: `studentInfo`, `studentRegistration`, `studentAssessment`, `assessments`, `courses`, `vle`, `studentVle` *(verify names)* | Convert `studentVle.csv` (~433 MB) to parquet once; how withdrawal is coded; how many students have no click rows at all |
| ③ P3 | CDC YRBS national data file (latest year + 1 earlier year for trend) + the **data user's guide**. Malaysia survey already in repo | File format (ASCII/SAS/SPSS/Access?) and how to load it in Python; names of the weight/strata/PSU variables; which questions measure sadness, sleep, screen time, bullying, activity, substance use; how "skipped" and "not asked" are coded |
| ④ P4 | Zenodo 3890931: the CSV + **both variable PDFs** | What one row means (one menu line per day? per canteen?); which columns mark base vs intervention weeks, meal type, canteen, customer group (student/staff, gender?) |
| ⑤ P5 | figshare collection 3893581: building power files, occupancy, **institute calendar**, building info, weather | Per-building file layout; timestamp format and timezone; known meter outage periods; which buildings are hostels vs academic; date range of weather (≈4 months only) |

### Deliverables
- 5 data cards; 5 profile reports; download script works for each pulse.

### Gate (team review, Day 3)
- Each person presents their data card in 3 minutes. Check: **is it real? is it big enough? is the target/question answerable?** If any pulse fails, switch it to the reserve (AISHE) now, not later.

---

## Phase 2 — Cleaning & signature challenge (Days 3–6)

**Goal:** clean data with a justified log of every step, and the "challenge" story for the presentation.

### Rules for everyone
- Never edit raw files. `clean(decisions)` is a pure function: raw → clean.
- Every step logs rows before/after and a one-line *why* via `DecisionLog`.
- For the **signature challenge**, compare **at least two handling options** and show a chart of how results differ. Make it a `DECISIONS` entry.
- Save output to `data/processed/<pulse>/clean.parquet`.

### Per-pulse tasks
**① Academic (P1)**
- [ ] Decode coded categories with the codebook; group rare (<1%) categories into "Other".
- [ ] **Signature: grades of 0 that are really missing.** Flag `enrolled>0 & grade==0` as `no_grade`; options `as_score | drop | nan_plus_flag | impute_plus_flag`. Argue missing-not-at-random: compare flagged vs graded students on dropout rate, age, debt (χ² / z).
- [ ] "Enrolled" outcome options: `exclude | as_non_dropout | own_class`.
- [ ] 0-enrolled-unit group (180 students): look at the credited columns, then decide and log.
- [ ] Features: `approval_rate`, `grade_change`, `financial_risk`, `cohort_id` (from the 10 unemployment/inflation/GDP combinations).

**② Engagement (P2)**
- [ ] Aggregate `studentVle` to **student × week** click totals (and per activity type); store as parquet.
- [ ] **Signature: noisy, spiky click logs + withdrawal leakage.** Remove clicks after the unregistration date; options for smoothing (`raw | rolling_mean_3w | ewm`); options for missing assessment scores (`drop | zero | median | flag`).
- [ ] Align course runs on "weeks since start" so different runs can be compared.
- [ ] Handle the missing deprivation band (`imd_band`): check whether missingness depends on region (missing at random?).

**③ Wellbeing (P3)**
- [ ] Load YRBS with `pyreadstat`/`pandas.read_sas` (or per the user's guide); keep the ~15–25 relevant questions.
- [ ] **Signature: skipped sensitive questions.** Compare skip rates across groups (sex, grade, race/ethnicity); options `complete_case | impute_mode | missing_as_category`; discuss missing-at-random vs not-at-random.
- [ ] Survey weights: compute key prevalences **weighted and unweighted** and show the difference (a decision toggle).
- [ ] Malaysia survey: **text cleaning** of the 49 course spellings (lowercase → strip → abbreviation map → `rapidfuzz` match); before/after table; parse CGPA ranges; the year-of-study casing fix (reuse the legacy notebook).

**④ Food (P4)**
- [ ] Reshape to one row per date × canteen × menu line (or whatever the variable PDF says is the right unit).
- [ ] Label each date: week number, base vs intervention, weekday, holidays/events.
- [ ] **Signature: demand outliers vs real events.** Detect spikes/drops (IQR on residuals after removing weekday effect); options `keep | winsorize | drop`; check each against the calendar.
- [ ] Smoothing: raw vs 7-day rolling vs weekday-adjusted sales (Lecture 5).

**⑤ Energy (P5)**
- [ ] Parse timestamps, set timezone, resample 1-min → **15-min and hourly** (store parquet).
- [ ] **Signature: meter gaps and spikes.** Find gaps (length distribution); options `drop_day | linear_interp | same_weekday_profile`; spikes via rolling z-score/IQR, then mark each against the calendar (exam, break, festival) and occupancy.
- [ ] Join occupancy (10-min) and the institute calendar; build the **day × hour profile matrix** (rows = days, columns = 24 hourly values) per building (Lecture 11).
- [ ] Label buildings: hostel vs academic vs other.

### Deliverables
- `clean.parquet` + `DecisionLog` table per pulse; one "before vs after" figure per pulse for the slides.

### Gate — mid-point sync (Day 6)
- Each person: 1 slide, 2 minutes, "**what was messy, what options we tried, what we chose and why**". The team challenges each choice as an examiner would.

---

## Phase 3 — Statistics & hypothesis testing (Days 5–8)

**Goal:** describe the data honestly and test 3+ specific hypotheses per pulse.

### Rules
- Use `common/stats_tests.py` only, so all tests report the same way.
- Every test: hypotheses written in words → test → statistic → p → **effect size** → Bonferroni-adjusted decision → one-sentence plain-English meaning.
- Check assumptions (sample size, expected counts ≥ 5, independence) and say which ones fail.

### Per-pulse hypotheses (starting points; refine after Phase 2)
| Pulse | H1 | H2 | H3 |
|---|---|---|---|
| ① | Dropout ⟂ debtor? (χ²) | Admission grade: dropout vs graduate (z) | **Debt effect within each semester-1 approval-rate band** (stratified χ²) |
| ② | Week-1–4 clicks: withdrawn vs passed (z/t) | Withdrawal ⟂ deprivation band (χ²) | Variance of clicks: distinction vs fail (F) |
| ③ | Sadness ⟂ <8 h sleep (χ²) | Sadness ⟂ 3+ h screen time (χ²) | Malaysia: depression ⟂ anxiety (χ², reuse) |
| ④ | Mean daily sales: base vs intervention weeks (t) | Meat share by customer group (χ²) | Sales variance base vs intervention (F) |
| ⑤ | Night-time hostel load: exam vs normal weeks (t) | Weekday vs weekend academic load (t) | Load variance: semester vs vacation (F) |

### Also per pulse
- [ ] One-variable summaries (distribution shape, skew, spread) and two-variable views (scatter/box/heatmap) for key variables.
- [ ] **Causation note:** only ④ has a randomised design. Everyone else writes "association, not causation".

### Deliverables
- `notebooks/<pulse>/03_statistics.ipynb` + a results table (`reports/<pulse>/tests.csv`).

### Gate
- Every test row has an effect size and a corrected decision.

---

## Phase 4 — Dimensionality reduction (Days 7–9)

**Goal:** find the main structure in each dataset and explain it.

| Pulse | What to reduce | Methods | What to look for |
|---|---|---|---|
| ① | Numeric + one-hot categorical student features | PCA (numeric), **SVD** (one-hot), t-SNE coloured by outcome | Do dropouts separate? Does the answer change between what's known at enrolment, after semester 1 and after semester 2? |
| ② | Student × week click matrix (first 10 weeks) | PCA of trajectories, t-SNE | Do the main components look like "early drop-off" or "steady" shapes? |
| ③ | Binary behaviour answers | PCA / SVD on the 0/1 matrix, t-SNE | Are there behaviour axes (e.g., "screen and sleep" vs "substance use")? |
| ④ | Day × menu-line sales matrix | PCA | Do intervention weeks separate from base weeks? |
| ⑤ | Day × 24-hour profile matrix | PCA (load shapes as components), **autoencoder** (reconstruction error = unusual day) | PC1 ≈ overall level, PC2 ≈ day vs night shift? Do exam days sit apart? |

- [ ] Each: explained-variance plot + loadings/biplot + a one-line interpretation per component.
- [ ] Scale features before PCA and log that choice as a decision.
- [ ] Use a fixed `random_state` for t-SNE; try 2 perplexities and say whether the picture changes.

### Gate
- Every plot has a written interpretation in the notebook and a caption ready for the dashboard.

---

## Phase 5 — Association rules & clustering (Days 8–11)

**Goal:** find combinations of attributes that occur together, and natural groups.

### Association rules (`common/rules.py` → mlxtend Apriori)
| Pulse | Transactions = | Example target rules |
|---|---|---|
| ① | Each student, with binned attributes | `{fees_late, no_scholarship, low_sem1_approval} → Dropout` |
| ② | Each student: demographic bands + early-activity bands | `{low_week1-4_clicks, no_assessment_1} → Withdrawn` |
| ③ | Each respondent: behaviour flags | `{<8h_sleep, 3h+_screen, bullied} → persistent_sadness` |
| ④ | Each day × line: meal attributes + customer group + week type | `{intervention_week, vegan} → low_sales?` |
| ⑤ | Each day: calendar + occupancy band + weather band | `{exam_week, hostel} → high_night_load` |

- [ ] Report support, confidence and **lift**; filter on lift > 1.2 and a minimum support appropriate to the dataset; show the top 10 with interpretation.
- [ ] Dashboard: sliders for min support / min confidence.

### Clustering (`common/clustering.py`)
- [ ] Every pulse: **K-Means + at least one of** Agglomerative (dendrogram), DBSCAN (k-distance plot), OPTICS (reachability).
- [ ] Choose k / eps with evidence (elbow, silhouette, dendrogram) and compare methods with silhouette, Davies–Bouldin and adjusted Rand index (ARI).
- [ ] **Name every cluster** from its profile and report the outcome rate per cluster.
- [ ] Profile **DBSCAN noise points** separately. Who doesn't fit any group?

| Pulse | What gets clustered | Example group names |
|---|---|---|
| ① | Students | "Financially stretched", "Mature evening student", "Strong starter", "Slipping after sem 1" |
| ② | Weekly click trajectories | "Steady", "Early fade", "Last-minute", "Never started" |
| ③ | Behaviour profiles | "Sleep-deprived screen-heavy", "Active and rested", … |
| ④ | Days / menu lines | "High-demand meat day", "Quiet veg day", … |
| ⑤ | Daily load shapes | "Normal teaching day", "Holiday", "Exam night-owl", "Meter fault" |

### Gate
- Each pulse has ≥3 interpreted rules and named clusters, with outcome rates.

---

## Phase 6 — Robustness & cross-pulse links (Days 10–12)

**Goal:** the shared contribution — show which findings survive different cleaning choices, and connect the five pulses.

### Robustness Matrix (everyone)
- [ ] Fill `FINDINGS` with your 3–5 headline findings, each as a number (e.g., "dropout-rate gap, debtor vs not").
- [ ] `run_pipeline(decisions)` recomputes them from scratch for any combination of decisions.
- [ ] `robustness_matrix(...)` → table + heatmap: finding × decision option → holds / weakens / reverses.
- [ ] Write one sentence: "Our most fragile finding is ___ because ___."

### Cross-pulse links (owners in brackets)
- [ ] **Campus Calendar** [P5 leads, all contribute]: a shared `calendar_events` table (term start, exams, breaks) and each pulse's key signal plotted on a "week of term" axis.
- [ ] **①↔② Do dropout warning signs carry over between institutions?** [P1 + P2]: a table of matched signals (hardship ↔ deprivation band, early engagement ↔ week-1–4 clicks, age ↔ age band, prior education ↔ highest education), with effect sizes side by side.
- [ ] **③↔⑤ Hostel late-night energy at exam time** [P3 + P5]: hostel 00:00–05:00 load, exam vs normal weeks (from Phase 3 H1), shown next to the YRBS sleep finding. Label it clearly as **indirect**.
- [ ] **③↔④ Food and wellbeing** [P3 + P4]: YRBS diet questions vs canteen choice patterns (a descriptive comparison only).
- [ ] **All:** "How fragile are campus insights?" Combine the five matrices into one view.

### Gate
- 5 Robustness Matrices render from the shared helper; ≥3 cross-pulse links have a chart.

---

## Phase 7 — Dashboard integration (Days 9–13)

**Goal:** one coherent app with ≥5 working functionalities per pulse page and two cross-pulse pages.

### Page template (every pulse page uses the same tab structure)
1. **Problem & data card** (real-data badge)
2. **Cleaning**: decision log table + signature-challenge toggle with a live before/after chart
3. **Statistics**: descriptives + hypothesis-test picker
4. **Structure**: PCA/SVD/t-SNE/autoencoder view
5. **Patterns**: rules with sliders + clusters with group cards
6. **Robustness**: this pulse's matrix

### Shared pages
- `Home.py`: "Campus Health Report", five cards (headline number + traffic-light status + link).
- `6_Campus_Calendar.py`: all pulses on one term timeline.
- `7_How_Fragile_Are_Insights.py`: combined robustness view.

### Engineering checklist (P1 leads)
- [ ] `@st.cache_data` on loaders; **precompute** t-SNE, the autoencoder and the I-BLEND/OULAD aggregates to parquet, so the app never touches raw 1-min/click files.
- [ ] App loads in < 10 s from a fresh start on a laptop.
- [ ] Each page works when its pulse's raw data is missing (shows "run download script").
- [ ] **Feature freeze at end of Day 12.** After that, only bug fixes.

### Gate
- A teammate who didn't build a page can use it without explanation.

---

## Phase 8 — Story, slides & Q&A (Days 13–15)

**Goal:** a 7-minute story that scores on all five equally weighted criteria.

- [ ] **Script** (see PLAN.md §8 for the time split): motivation → five pulses (one insight + one challenge each, ~45 s, live in the dashboard) → cross-pulse links → robustness → limitations/implications/next steps.
- [ ] **Slides:** ≤ 1 per pulse plus intro, links, robustness, limitations (≈10 total). Each pulse slide uses the same layout: *problem · messy data · what we did · one insight*.
- [ ] **Backup:** a 90-second screen recording of the dashboard demo in case the live demo fails.
- [ ] **Q&A bank:** 5 questions per person about their own pulse + the shared ones in PLAN.md §8. Each answer ≤ 30 seconds. Practise by asking each other unprepared questions.
- [ ] **Limitations / implications / future work slide:** taken from each pulse's findings notebook.
- [ ] Two full **timed rehearsals**; cut content until it's ≤ 7:00 with 15 s to spare.

### Gate
- Both rehearsals ≤ 7:00; every member has answered at least 3 unprepared questions in practice.

---

## Phase 9 — Final freeze & submission (Days 15–16)

- [ ] `README.md`: one-paragraph summary, setup, download, run notebooks, run app, team & pulse ownership.
- [ ] **Fresh-clone test** on a different laptop: clone → install → download → run notebooks → run app.
- [ ] All notebooks run top to bottom without errors; outputs stripped except where we want figures kept.
- [ ] `reports/` contains final figures and slides (PDF).
- [ ] Tag `v-presentation` on `main`.
- [ ] Submit whatever the course form asks for (team details, link, slides).

---

## Weekly rhythm

| When | What | Length |
|---|---|---|
| Daily | Async update in the group chat: done / next / blocked | 3 lines |
| Day 1 | Contracts call (Phase 0) | 30 min |
| Day 3 | Data-card review (Phase 1 gate) | 20 min |
| Day 6 | Mid-point sync (Phase 2 gate) | 30 min |
| Day 9 | Patterns review: DR, rules, clusters first look | 30 min |
| Day 12 | Feature freeze + dashboard walkthrough | 30 min |
| Day 14–15 | Rehearsals | 2 × 20 min |

## Minimum viable version per pulse (if someone falls behind)
Clean data + decision log · 2 hypothesis tests · PCA · one rules table · K-Means with named clusters · one dashboard page with one toggle. This still meets the "≥5 functionalities" requirement. Everything else is extra.
