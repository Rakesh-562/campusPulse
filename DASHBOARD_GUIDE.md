# CampusPulse Dashboard Guide

How to use the dashboard to answer each research question, what every tab shows, and what to say about it.

---

## 1. What the dashboard is

The dashboard answers three research questions about a university, one per real dataset, and shows for every
answer whether it survives a different data-cleaning choice.

- **Open it:** double-click `campusPulse/dashboard/index.html`. No server, install or internet is needed.
- **Where the numbers come from:** `scripts/export_dashboard_data.py` runs the three pipelines and writes
  `dashboard/data.js`. The page only displays that file, so it can never disagree with the code or the notebooks.
  After changing a pipeline, run the export again and reload the page.

| Page (top bar) | What it is for |
|---|---|
| Overview | The umbrella question and the three research questions with their datasets |
| ① Academic, ② Wellbeing, ③ Energy | One page per research question |
| Robustness | Every finding recomputed under every alternative cleaning choice |
| Conclusions | One block of conclusions per question, the answer to the umbrella question, and the limitations |

Each research-question page has six numbered tabs, in the order we present them. Only the chosen tab is shown:

| Tab | What it shows |
|---|---|
| 1. Data processing | The main data problem of that dataset, its graphs, and the step-by-step cleaning log |
| 2. Statistics | Descriptive statistics and one simple hypothesis test per finding |
| 3. Dimensionality reduction | PCA and t-SNE maps coloured by outcome, variance explained, loadings |
| 4. Association rules | Two Apriori rules per pulse that sound like common sense but have lift below 1 |
| 5. Clustering | K-Means groups on a PCA map and group profiles |
| 6. Findings | Every finding with its own graph and test, the chart that answers the question, and the cleaning-choice selector |

The address bar remembers the view: `index.html#/energy/findings` opens the Energy page on Findings.
Put these links in your slides to jump straight to a view.

---

## 2. How to read every page

**Hypothesis tests.** Only the simple tests from the course are used:

| Comparing | Test | Used for |
|---|---|---|
| Two percentages (e.g. dropout rate of debtors vs non-debtors) | Two-proportion z-test | A1, A2, A4, W1, W2, W3 |
| Two means (e.g. mean grade, mean power) | Two-sample t-test (Welch) | A3, EN1, EN2 |

Each test is written as H0, the statistic, p and the decision at 5%. With thousands of rows almost every test
rejects H0, so the **size** of the difference (shown on each finding's graph) matters more than p.

**Cleaning-choice selector** (Findings tab, bottom). Each dropdown is one cleaning decision; the first option,
marked "(default)", is ours. Picking another recomputes every finding with only that decision changed.

**Status labels** (coloured dot plus a word):

| Status | Meaning |
|---|---|
| holds | Same direction, and between half and double our default value |
| weakens | Same direction, but shrinks below half |
| inflates | Same direction, but more than double: the cleaning choice creates most of the effect (a warning) |
| reverses | The direction flips: the claim no longer holds |
| n/a | Cannot be computed under that choice |

**Units.** "pp" = percentage points, the plain difference between two percentages: 75.5% − 34.5% = +41.1 pp.
Energy is in kilowatts (kW).

**Charts.** Hover any bar or dot for its exact value. On finding graphs, orange and blue are the two groups being
compared (e.g. debtors and non-debtors); red marks a result that fails.

---

## 3. ① Academic: Who drops out, and how early can we tell?

**Answer:** simple rules on resolved student records can support early, selective outreach without regression.
At enrolment, a fee/debt check flags **18.5%** of students and catches **39%** of future dropouts. By the end of
semester 1, a course-progress rule combined with unpaid fees flags **27.8%** and catches **68%**; **95%** of
those flagged students later drop out. A semester-2 course rule catches 66%, so waiting adds no benefit.
*Data: UCI 697, 4,424 students at a Portuguese polytechnic; 3,630 with a known outcome.*

| Tab | What to look at | What it tells you |
|---|---|---|
| 1. Data processing | Challenge tiles (hidden zeros, uncredited units), raw semester-1 grade histogram, cleaning log | 538 students were enrolled but graded 0; 91.6% of them dropped out against 23.9% of others, so a 0 is a missing exam, not a score. The raw-grade histogram shows the spike at 0. Treating 0 as a score inflates the grade gap 5.5×. Auditable cleaning log records before/after counts for every cleaning decision. |
| 2. Statistics | Cohort outcome breakdown, dropout vs non-dropout by debt and scholarship status, age histogram, descriptive table, test table, debt within approval bands chart | Visual overview of the full cohort (32.1% dropout, 49.9% graduate, 17.9% enrolled), outcome shares among resolved students by debt and scholarship status, age distribution, descriptive statistics for key features, tests, and debt vs dropout across approval bands |
| 3. Dimensionality reduction | PCA and t-SNE | Dropouts and graduates separate on their own; PC1 is course progress |
| 4. Association rules | Two common beliefs the data contradicts | Living away from home → dropout: lift 0.85 (33% vs 39% overall). International student → dropout: lift 0.91 |
| 5. Clustering | Group profiles | One group of 232 students drops out at 90.5%; the largest group (709) at 23.6% |
| 6. Findings | Four finding graphs and cleaning-choice selector | See below |

| Finding | Graph shows | Simple test |
|---|---|---|
| A1 Debtors drop out more: **+41.1 pp** | Dropout rate 75.5% (debtors) vs 34.5% (non-debtors) | Two-proportion z = 16.10, p < 0.0001 → reject H0 |
| A2 Scholarship holders drop out less: **+34.5 pp** | 48.4% without vs 13.8% with a scholarship | z = 18.86, p < 0.0001 → reject H0 |
| A3 Dropouts have lower semester-1 grades: **+0.98 points** | Mean grade 13.1 (persisted) vs 12.1 (dropped out), out of 20 | Welch t = 18.41, p < 0.0001 → reject H0 |
| A4 Debt matters within grade bands: **+23.8 pp** | Debt gap +1.9 pp (failing band), +48.1 pp (middle), +21.3 pp (passed everything) | z-test per band: failing band z = 1.59, p = 0.11 → **not** significant; middle z = 10.31 and passed z = 6.98 → reject H0 |

**Say:** debt does not add risk for students who are already failing; it matters most in the middle band.

**Try live:** set *zero grade* to *as score*: A3 turns orange (inflates).

**Caveats:** one institution in Portugal; association, not cause; 794 students with an unresolved outcome are excluded.

---

## 4. ② Wellbeing: Which behaviours co-occur with persistent sadness?

**Answer:** bullying has the strongest link (+34.1 pp), then short sleep (+18.9 pp), then inactivity (+11.5 pp).
All three hold under every cleaning choice. *Data: Student Wellbeing Survey 2023, national sample, 20,103 respondents;
13,322 complete cases after requiring answers to the outcome and all indicator inputs. Rates are survey-weighted.*

| Tab | What to look at | What it tells you |
|---|---|---|
| 1. Data processing | Missing-answer chart and derived-indicator guide | Shows which responses are recoded into sadness, sleep, bullying, inactivity, heavy social-media and sex indicators |
| 2. Statistics | Weighted sleep and activity distributions, indicator prevalence, descriptive table and test table | Understands the retained sample and the five weighted indicator rates; all three finding tests reject H0 |
| 3. Dimensionality reduction | PCA (t-SNE not shown) | Variance is spread evenly, so PCA cannot compress the five yes/no features; with only 32 answer combinations t-SNE cannot be interpreted, so it is not shown. Overlapping sadness-outcome markers at the same PCA coordinate are offset horizontally for visibility |
| 4. Association rules | Two common beliefs the data contradicts | Heavy social media + 8 h sleep → sadness: lift 0.72 (29% vs 41%). Never exercises + male → sadness: lift 0.81 |
| 5. Clustering | K-Means groups on a PCA map and group profiles | Groups are combinations of yes/no answers; compare their sadness rates rather than treating them as hidden student types |
| 6. Findings | Three finding graphs + sadness by hours of sleep | See below |

| Finding | Graph shows | Simple test |
|---|---|---|
| W1 Sleep-deprived (< 8 h) report more sadness: **+18.9 pp** | 44.9% vs 26.0% | Two-proportion z = 18.64, p < 0.0001 → reject H0 |
| W2 Bullied students report more sadness: **+34.1 pp** | 65.9% vs 31.9% | z = 34.54, p < 0.0001 → reject H0 |
| W3 Inactive students report more sadness: **+11.5 pp** | 50.3% vs 38.8% | z = 8.02, p < 0.0001 → reject H0 |

The sleep chart: sadness falls from 60.5% at ≤4 hours to 25.4% at 8 hours, then rises again to 32.5% at 10+ hours.

**Try live:** set *survey weights* to *unweighted* or *sleep cutoff* to *less than 6 hrs*: W1 to W3 still hold.

**Caveats:** association, not cause; self-reported survey responses. Complete-case filtering leaves
13,322 of 20,103 respondents. Rates are weighted; the z-tests use sample counts and are not survey-design-adjusted.

---

## 5. ③ Energy: When does a campus building behave differently, and why?

**Answer:** the academic building draws **17.9 kW more** on teaching days, and the hostels **12.2 kW more at night**
in term than in vacation. *Data: I-BLEND, IIIT Delhi, 38,520 hours (Aug 2013 – Dec 2017), institute calendar.*

| Tab | What to look at | What it tells you |
|---|---|---|
| 1. Data processing | Meter-gap chart | The hostel meters sent no reading in about 25% of hours; by default those hours are dropped |
| 2. Statistics | Descriptive table, two-test table, **daily academic-building load curve** | The curve shows hourly mean power on teaching vs low-activity days |
| 3. Dimensionality reduction | PCA of daily load shapes | PC1 explains 57%: mostly the early-morning hours |
| 4. Association rules | Two common beliefs the data contradicts | Weekend → hostels busy: lift 0.83 (28% vs 33%). June, the hottest month → campus busy: lift 0.88 (summer vacation) |
| 5. Clustering | Day groups | 426 mostly weekend / vacation days (~24 kW at noon), 408 teaching days (~51 kW), 77 high-load teaching days (~57 kW) |
| 6. Findings | Two finding graphs + calendar note | The calendar distinguishes teaching from low-activity days, but does not identify exam dates |

| Finding | Graph shows | Simple test |
|---|---|---|
| EN1 Academic building uses more on teaching days: **+17.9 kW** | 45.5 kW vs 27.7 kW (9:00–18:00) | Welch t = 63.21, p < 0.0001 → reject H0 |
| EN2 Hostels use more at night in term: **+12.2 kW** | 54.5 kW vs 42.3 kW (00:00–05:00) | Welch t = 31.63, p < 0.0001 → reject H0 |

**Caveats:** the calendar has no exam dates, so we make no exam claims; hostel power is not a measurement of sleep.

---

## 6. The umbrella question: how fragile are the insights?

Open **Robustness**: each pulse is a card, rows = findings, columns = alternative cleaning choices.

| Pulse | Checks that hold | What moves |
|---|---|---|
| ① Academic | 15 of 16 | A3 inflates 5.5× if ungraded zeros are treated as scores |
| ② Wellbeing | 15 of 15 | Nothing moves |
| ③ Energy | 10 of 10 | Both displayed findings hold under every cleaning choice |

**The message:** most findings are solid, but a reasonable-looking choice (treating a missing exam as a 0, skipping
text cleaning) can create most of an effect, so cleaning decisions must be reported next to the numbers.

---

## 7. Presenting it

**7-minute demo**

| Time | Where | What to show |
|---|---|---|
| 0:00–1:00 | Overview | The umbrella question and the three questions |
| 1:00–5:00 | Each pulse, tabs 1 → 6 | Data processing first (the data problem), then one or two tabs, then Findings with its graphs and tests |
| 5:00–6:00 | Robustness | The cleaning choices that change an answer |
| 6:00–7:00 | Conclusions | Conclusions per question, the umbrella answer, limitations |

**Likely questions**

| Question | Short answer |
|---|---|
| Why is a 0 grade missing, not a low score? | 91.6% of those students dropped out and real grades start near 10: a 0 is a missed exam |
| Which test did you use, and why? | z-test for two percentages, t-test for two means, chi-square for two categories: the simplest test for each comparison |
| Everything has p < 0.0001: does that mean the effects are big? | No: with thousands of rows tiny differences are significant, so we show the size of each difference on its graph |
| Is debt a cause of dropout? | Not proven: an association. A4 shows it matters even among students with similar grades, except those already failing |
| Why no exam analysis? | The I-BLEND calendar has no exam dates |
| What is pp? | Percentage points: the difference between two percentages, e.g. 75.5% − 34.5% = 41.1 pp |
| What is lift? | How many times the overall rate the "then" happens when the "if" is true; above 1 = more often, below 1 = less often. Our rules tab shows beliefs with lift below 1 |
| How do you know the first dataset was synthetic? | Every row's five "hours per day" add up to exactly 24, and a depth-3 tree reproduces "stress" 100% |
