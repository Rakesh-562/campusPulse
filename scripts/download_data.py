"""Download the raw data for each pulse into data/raw/<pulse>/.

Usage:
    python scripts/download_data.py --pulse all            # or: academic, iblend, yrbs
Then build the processed tables:
    python scripts/preprocess_yrbs.py
    python scripts/preprocess_iblend.py
"""
import argparse
import zipfile
from pathlib import Path
import requests

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def download(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  already here: {dest.name}")
        return
    print(f"  downloading {dest.name} ...")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)


def unzip(zip_file: Path):
    with zipfile.ZipFile(zip_file) as z:
        z.extractall(zip_file.parent)
    print(f"  unpacked {zip_file.name}")


def academic():
    """UCI 697, already committed in data/raw/uci_dropout/."""
    out = RAW / "uci_dropout" / "data.csv"
    if out.exists():
        print("  already here: uci_dropout/data.csv")
        return
    z = RAW / "uci_dropout" / "dropout.zip"
    download("https://archive.ics.uci.edu/static/public/697/predict+students+dropout+and+academic+success.zip", z)
    unzip(z)


def iblend():
    """I-BLEND (IIIT Delhi): energy (~590 MB zip, 1.6 GB unpacked), occupancy, calendar, weather."""
    articles = requests.get("https://api.figshare.com/v2/collections/3893581/articles",
                            params={"page_size": 100}, timeout=60).json()
    for article in articles:
        for f in requests.get(article["url"], timeout=60).json()["files"]:
            z = RAW / "iblend" / f["name"]
            download(f["download_url"], z)
            if z.suffix == ".zip" and not (RAW / "iblend" / z.stem).exists():
                unzip(z)


def yrbs():
    """CDC YRBS 2023 national high-school data (fixed-width) plus the SAS programs that describe it."""
    base = "https://www.cdc.gov/yrbs/files/2023/"
    for name in ["XXH2023_YRBS_Data.dat", "2023XXH-SAS-Input-Program.sas", "2023XXH-Formats-Program.sas"]:
        download(base + name, RAW / "yrbs" / name)


JOBS = {"academic": academic, "iblend": iblend, "yrbs": yrbs}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pulse", default="all", choices=[*JOBS, "all"])
    args = parser.parse_args()
    for name in (JOBS if args.pulse == "all" else [args.pulse]):
        print(f"[{name}]")
        JOBS[name]()
