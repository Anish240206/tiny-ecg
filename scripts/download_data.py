"""Download MIT-BIH Arrhythmia Database (~100 MB) into data/mitdb.
Run from repo root:  python scripts/download_data.py"""
from pathlib import Path
import wfdb

dest = Path(__file__).resolve().parents[1] / "data" / "mitdb"
dest.mkdir(parents=True, exist_ok=True)
wfdb.dl_database("mitdb", str(dest))
print("Done ->", dest, "| files:", len(list(dest.iterdir())))
