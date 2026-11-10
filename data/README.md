# CampusPulse — Data Cards

This directory documents the data sources, licenses, schemas, time spans, real-or-synthetic evidence, and known challenges for each vital sign.

---

## 1. Academic Pulse: UCI 697 Predict Students' Dropout and Academic Success
- **Source**: UCI Machine Learning Repository (Dataset ID 697) / Polytechnic Institute of Portalegre, Portugal.
- **Citation**: Realinho, V., Machado, J., Baptista, L., & Martins, M. V. (2022). *Predict Students' Dropout and Academic Success*.
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Format & Size**: CSV, ~520 KB (`data/raw/uci_dropout/data.csv`).
- **Dimensions**: 4,424 rows × 37 columns.
- **Time Span**: Longitudinal records across multiple academic enrollment cohorts (2008–2019).
- **Real-or-Synthetic Evidence**: Real institutional higher-education data. Contains real administrative codes, tuition payment histories, age distributions with natural skew, and authentic semester credit evaluations.
- **Known Challenges & Quirks**:
  1. *Hidden Zeros*: 538 students have a semester-1 grade of exactly `0.0`. 92% of these students dropped out. These zeros represent non-attendance/missing exams rather than evaluated academic performance (Missing Not At Random).
  2. *Right-Censored Outcomes*: 794 students are categorized as "Enrolled" (outcome not yet resolved at time of study).
  3. *Cohort Markers*: Macroeconomic indicators (unemployment rate, inflation rate, GDP) are identical across students in the same year, functioning as cohort labels rather than individual economic status.

---

## 2. Engagement Pulse: OULAD (Open University Learning Analytics Dataset)
- **Source**: Knowledge Media Institute, The Open University, United Kingdom.
- **Citation**: Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). *Open University Learning Analytics dataset*. Scientific Data, 4:170171.
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Format & Size**: Multi-table relational dataset (7 CSVs); `studentVle.csv` click logs is ~433 MB uncompressed (`data/raw/oulad/`).
- **Dimensions**: 32,593 unique students, 22 course presentations across 7 modules; 10,655,280 VLE click events.
- **Time Span**: 2013–2014 academic course presentations (February "B" and October "J" cohorts).
- **Real-or-Synthetic Evidence**: Real VLE server interaction logs. Exhibits realistic diurnal, weekly, and assessment deadline burst patterns, long-tailed power-law activity distributions, and student dropout drop-off.
- **Known Challenges & Quirks**:
  1. *Volume & Granularity*: 10+ million interaction records must be resampled and aggregated into weekly engagement matrices (`data/processed/engagement/weekly_clicks.parquet`).
  2. *Extreme Activity Spikes*: Assessment submission dates generate sharp, noisy click spikes requiring smoothing (e.g. rolling averages vs exponential smoothing).
  3. *Missing Assessment Grades*: Students who disengage early do not submit later assessments.

---

## 3. Wellbeing Pulse: CDC YRBS + Malaysia Student Mental Health Survey
- **Source 1 (CDC YRBS)**: Centers for Disease Control and Prevention (CDC) Youth Risk Behavior Surveillance System (2023/2025 release).
- **Source 2 (Malaysia Survey)**: Public University Student Mental Health Survey (Malaysia), Kaggle.
- **Citation**: Centers for Disease Control and Prevention. *Youth Risk Behavior Surveillance System (YRBSS)*.
- **License**: Public Domain (CDC US Government Work) / Open Dataset.
- **Format & Size**:
  - YRBS: Microsoft Access MDB (35.5 MB) exported to processed CSV (`data/processed/wellbeing/yrbs_xxhqn.csv`, 10,710 respondents × 153 variables).
  - Malaysia: CSV (7.3 KB, 101 respondents × 11 variables).
- **Time Span**: Modern surveillance cohorts.
- **Real-or-Synthetic Evidence**: Real survey responses. Exhibits authentic skip patterns, item non-response on sensitive behavioral questions (suicidality, substance abuse, sexual violence), and authentic typographical variance.
- **Known Challenges & Quirks**:
  1. *Sensitive Question Non-Response*: Missing data is non-random; respondents with acute distress occasionally skip specific behavioral probes.
  2. *Unstructured Free-Text*: The Malaysia survey contains 49 different unstandardized spellings and abbreviations for degree courses (e.g. `engin`, `Engineering`, `KOE`, `KIRKHS`, `Laws`, `Law`), requiring fuzzy string matching and text normalization.

---

## 4. Food Pulse: NOVANIMAL Canteen Menu Sales Experiment
- **Source**: ZHAW Zurich University of Applied Sciences, Switzerland.
- **Citation**: Egeler, C., & Baur, I. (2020). *Aggregated menu sales data from two university canteens (NOVANIMAL)*. Zenodo.
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Format & Size**: CSV, ~5.8 MB (`data/raw/novanimal/2017_ZHAW_aggregated_menu_sales_NOVANIMAL.csv`).
- **Dimensions**: 26,340 transaction records across 2 university canteens (Grüental and Vista).
- **Time Span**: 12-week controlled field experiment during Spring semester 2017.
- **Real-or-Synthetic Evidence**: Real point-of-sale till transactions from university cafeterias. Contains verified sales amounts, Swiss Franc price tiers, weather metrics (rain, temperature, sunshine), and customer group tags (`Studierende`, `Mitarbeiter`).
- **Known Challenges & Quirks**:
  1. *Experimental Design*: Alternating Basis (baseline) and Intervention cycles where vegetarian/vegan buffet shares and prominence were expanded.
  2. *Strong Weekly Seasonality & Outliers*: Mondays and Fridays exhibit lower on-campus attendance; public holidays create zero-sales anomalies that must be distinguished from consumer rejection.
  3. *Aggregated Item Sales*: Data represents till aggregates per meal line rather than individual patron longitudinal tracking.

---

## 5. Energy Pulse: I-BLEND IIIT Delhi Smart Campus
- **Source**: Indraprastha Institute of Information Technology Delhi (IIITD), India.
- **Citation**: Rashid, H., Singh, P., & Singh, A. (2019). *I-BLEND: a campus-scale commercial and residential building electrical energy dataset*. figshare.
- **License**: figshare Academic Open Access / CC BY 4.0.
- **Format & Size**: 16 CSV files totaling ~1.2 GB; `all_buildings_power.csv` (~196 MB) covers 52 months of 1-minute power telemetry.
- **Dimensions**: 9 campus facilities (Academic, Boys Hostel mains/ups, Girls Hostel mains/ups, Lecture Hall, Library, Dining/Mess, Facilities).
- **Time Span**: August 2013 to December 2017 (52 months continuous telemetry).
- **Real-or-Synthetic Evidence**: Physical IoT smart meter telemetry with real power factor fluctuations, campus-wide grid power cuts (trips to UPS backup), and seasonal air-conditioning load shifts.
- **Known Challenges & Quirks**:
  1. *High-Frequency Volume & Meter Gaps*: 1-minute sampling yields millions of data points with occasional communication timeouts and power cuts.
  2. *Meter Spikes vs True Events*: Severe transient spikes (sensor glitches) must be separated from genuine high-load events (commencement, tech fests, heat waves).
  3. *Temporal Resampling & Alignment*: 1-min raw watts must be resampled to hourly and daily energy profiles, aligned with academic exam calendars and hostel night schedules (00:00–05:00).
