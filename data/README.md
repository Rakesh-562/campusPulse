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

## 2. Wellbeing Pulse: Student Health & Wellbeing Survey 2023
- **Source 1**: National Student Health & Wellbeing Behavioral Survey (2023/2025 release).
- **Citation**: National Student Behavioral and Mental Health Survey.
- **License**: Public Domain / Open Dataset.
- **Format & Size**:
  - YRBS: Microsoft Access MDB (35.5 MB) exported to processed CSV (`data/processed/wellbeing/yrbs_xxhqn.csv`, 10,710 respondents × 153 variables).
- **Time Span**: Modern surveillance cohorts.
- **Real-or-Synthetic Evidence**: Real survey responses. Exhibits authentic skip patterns, item non-response on sensitive behavioral questions (suicidality, substance abuse, sexual violence), and authentic typographical variance.
- **Known Challenges & Quirks**:
  1. *Sensitive Question Non-Response*: Missing data is non-random; respondents with acute distress occasionally skip specific behavioral probes.

---

## 3. Energy Pulse: I-BLEND IIIT Delhi Smart Campus
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
