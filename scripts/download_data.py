"""Usage: python scripts/download_data.py --pulse {academic,oulad,novanimal,iblend,yrbs,all}"""
import argparse, zipfile
from pathlib import Path
import requests

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

def _download(url, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  skip (exists): {dest.name}"); return
    print(f"  downloading {dest.name}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)

def academic():
    out = RAW / "uci_dropout"
    if any(out.glob("*.csv")):
        print("  UCI dropout already present"); return
    from ucimlrepo import fetch_ucirepo
    fetch_ucirepo(id=697).data.original.to_csv(out / "data.csv", index=False)

def oulad():
    out = RAW / "oulad"
    z = out / "anonymisedData.zip"
    _download("https://analyse.kmi.open.ac.uk/open_dataset/download", z)
    with zipfile.ZipFile(z) as zf:
        zf.extractall(out)

def novanimal():
    meta = requests.get("https://zenodo.org/api/records/3890931", timeout=60).json()
    for f in meta["files"]:
        _download(f["links"]["self"], RAW / "novanimal" / (f.get("key") or f.get("filename")))

def iblend():
    arts = requests.get("https://api.figshare.com/v2/collections/3893581/articles",
                        params={"page_size": 100}, timeout=60).json()
    for a in arts:
        info = requests.get(a["url"], timeout=60).json()
        for f in info["files"]:
            _download(f["download_url"], RAW / "iblend" / f["name"])

def yrbs():
    print("  YRBS must be downloaded manually: https://www.cdc.gov/yrbs/data/index.html\n"
          "  Put the file(s) in data/raw/yrbs/ (read with pyreadstat / pandas.read_sas).")

JOBS = {"academic": academic, "oulad": oulad, "novanimal": novanimal, "iblend": iblend, "yrbs": yrbs}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pulse", default="all", choices=[*JOBS, "all"])
    a = p.parse_args()
    for name in (JOBS if a.pulse == "all" else [a.pulse]):
        print(f"[{name}]"); JOBS[name]()
