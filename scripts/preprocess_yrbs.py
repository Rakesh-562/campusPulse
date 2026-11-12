"""Build data/processed/wellbeing/yrbs_2023.csv from the CDC 2023 national YRBS files.

Input  (python scripts/download_data.py --pulse yrbs):
    data/raw/yrbs/XXH2023_YRBS_Data.dat          fixed-width answers, one student per line
    data/raw/yrbs/2023XXH-SAS-Input-Program.sas   tells us where each question sits on the line

The question numbers change between YRBS years, so we give every column a plain name here and the
pipeline never uses question numbers. Question meanings come from the 2023 SAS label and format programs:
    Q2  sex                          1 = female, 2 = male
    Q3  grade                        1 = 9th ... 4 = 12th
    Q24 bullied at school            1 = yes, 2 = no
    Q25 bullied electronically       1 = yes, 2 = no
    Q26 sad or hopeless 2+ weeks     1 = yes, 2 = no
    Q76 days physically active 60+ min in the last 7 days     1 = 0 days ... 8 = 7 days
    Q80 social media use             1 = never ... 6 = several times a day, 7 = hourly, 8 = more than hourly
    Q85 hours of sleep on a school night   1 = 4 or less, 2 = 5, 3 = 6, 4 = 7, 5 = 8, 6 = 9, 7 = 10 or more
"""
import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "yrbs"
OUT = ROOT / "data" / "processed" / "wellbeing" / "yrbs_2023.csv"

WANTED = {"Q2": "sex", "Q3": "grade", "Q24": "bullied_school", "Q25": "bullied_online", "Q26": "sad",
          "Q76": "active_code", "Q80": "social_media_code", "Q85": "sleep_code",
          "WEIGHT": "weight", "STRATUM": "stratum", "PSU": "psu"}


def column_positions(sas_file: Path) -> dict:
    """Read lines like '@57 Q26 $1.' or '@388 WEIGHT 10.4' -> {name: (start, end)} (0-based, end exclusive)."""
    positions = {}
    for line in sas_file.read_text(encoding="latin-1").splitlines():
        m = re.match(r"\s*@(\d+)\s+(\w+)\s+\$?(\d+)\.", line)
        if m:
            start, name, width = int(m.group(1)) - 1, m.group(2).upper(), int(m.group(3))
            positions[name] = (start, start + width)
    return positions


def main():
    pos = column_positions(RAW / "2023XXH-SAS-Input-Program.sas")
    names = list(WANTED)
    df = pd.read_fwf(RAW / "XXH2023_YRBS_Data.dat", colspecs=[pos[n] for n in names], names=names,
                     dtype=str, encoding="latin-1")
    df = df.rename(columns=WANTED)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")       # blank = skipped question -> NaN

    # Turn codes into readable values (still NaN where the student skipped the question)
    df["sleep_hours"] = df["sleep_code"].map({1: 4, 2: 5, 3: 6, 4: 7, 5: 8, 6: 9, 7: 10})
    df["active_days"] = df["active_code"] - 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT}: {len(df):,} students")
    print("share of skipped answers per question:")
    print(df.drop(columns=["weight", "stratum", "psu"]).isna().mean().round(3).to_string())


if __name__ == "__main__":
    main()
