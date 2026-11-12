"""Build data/processed/energy/hourly_campus_power.csv from the raw I-BLEND files (IIIT Delhi).

Input  (python scripts/download_data.py --pulse iblend, then unzip into data/raw/iblend/):
    energy_dataset/all_buildings_power.csv        1-minute power (W) of 9 meters, UNIX timestamps
    iiitd_calender_schedule/calender_year_*.csv   per day: working_day (1/0) and activity (H = teaching, L = low)

Important: the institute calendar does NOT mark exam days. It only says whether a day was a high-activity
teaching day (H) or a low-activity day (L: weekends, holidays, vacations). We therefore compare teaching days
with low-activity days and term with vacation, and never claim anything about exams.

Output: one row per hour with the mean power of each meter, a few totals, and calendar flags.
A meter that sent no reading in an hour is left empty (that is the "meter gap" the pipeline handles).
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "iblend"
OUT = ROOT / "data" / "processed" / "energy" / "hourly_campus_power.csv"
HOSTEL = ["Boys_main", "Boys_backup", "Girls_main", "Girls_backup"]
VACATION_MIN_DAYS = 14          # a run of at least 14 low-activity days in a row = vacation


def calendar() -> pd.DataFrame:
    files = sorted((RAW / "iiitd_calender_schedule").glob("calender_year_*.csv"))
    cal = pd.concat(pd.read_csv(f) for f in files)
    cal["date"] = pd.to_datetime(cal["Date"]).dt.date
    cal["is_teaching"] = (cal["activity"] == "H").astype(int)
    # Label vacation: long unbroken runs of low-activity days
    run_id = (cal["is_teaching"] != cal["is_teaching"].shift()).cumsum()
    run_length = cal.groupby(run_id)["is_teaching"].transform("size")
    cal["is_vacation"] = ((cal["is_teaching"] == 0) & (run_length >= VACATION_MIN_DAYS)).astype(int)
    return cal[["date", "is_teaching", "is_vacation"]]


def main():
    power = pd.read_csv(RAW / "energy_dataset" / "all_buildings_power.csv", na_values=["NA"])
    power["datetime"] = pd.to_datetime(power["timestamp"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata")
    hourly = power.drop(columns="timestamp").set_index("datetime").resample("1h").mean()

    hourly["total_hostel_power"] = hourly[HOSTEL].sum(axis=1, min_count=len(HOSTEL))   # empty if any hostel meter is missing
    meters = [c for c in hourly.columns if c not in ("total_hostel_power",)]
    hourly["total_campus_power"] = hourly[meters].sum(axis=1, min_count=len(meters) - 2)
    hourly = hourly.reset_index()
    hourly["hour"] = hourly["datetime"].dt.hour
    hourly["is_weekend"] = (hourly["datetime"].dt.dayofweek >= 5).astype(int)
    hourly["is_late_night"] = (hourly["hour"] < 5).astype(int)
    hourly["date"] = hourly["datetime"].dt.date
    hourly = hourly.merge(calendar(), on="date", how="left")
    hourly = hourly.dropna(subset=["is_teaching"]).drop(columns="date")   # keep hours the calendar covers

    OUT.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(OUT, index=False)
    print(f"wrote {OUT}: {len(hourly):,} hours from {hourly['datetime'].min()} to {hourly['datetime'].max()}")
    print("share of hours with no reading, per meter:")
    print(hourly[meters].isna().mean().round(3).to_string())
    print("teaching-day hours:", f"{hourly['is_teaching'].mean():.1%}", "| vacation hours:", f"{hourly['is_vacation'].mean():.1%}")


if __name__ == "__main__":
    main()
