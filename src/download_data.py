"""
Download Bureau of Transportation Statistics (BTS) on-time performance data
and keep only Southwest Airlines (carrier code WN) flights.

Downloads month by month from --start up to the latest month BTS has published
(it stops automatically at the first month that isn't available yet).
Each month is cached in data/raw/, so re-running resumes where it left off.

Usage:
  python src/download_data.py                      # last 24 months up to the latest available
  python src/download_data.py --start 2024-07      # from July 2024 to the latest available
  python src/download_data.py --start 2026-04 --end 2026-06
Output:
  data/southwest_flights.csv
"""
import argparse
import io
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
URL = ("https://transtats.bts.gov/PREZIP/"
       "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{y}_{m}.zip")
KEEP = [
    "FlightDate", "Month", "DayofMonth", "DayOfWeek", "Reporting_Airline",
    "Origin", "Dest", "CRSDepTime", "CRSArrTime", "Distance",
    "ArrDelay", "ArrDel15", "Cancelled", "Diverted",
    "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay",
]


def month_range(start: str, end: str):
    cur = pd.Period(start, "M")
    last = pd.Period(end, "M")
    while cur <= last:
        yield cur.year, cur.month
        cur += 1


def fetch_month(year: int, month: int) -> bool:
    """Download one month to the cache. Returns False if BTS hasn't published it."""
    out = RAW / f"wn_{year}_{month:02d}.csv"
    if out.exists():
        print(f"{year}-{month:02d}: already downloaded, skipping")
        return True
    print(f"{year}-{month:02d}: downloading ...", flush=True)
    r = requests.get(URL.format(y=year, m=month), timeout=600)
    if r.status_code == 404:
        return False
    r.raise_for_status()
    if not zipfile.is_zipfile(io.BytesIO(r.content)):  # BTS sometimes returns an HTML page instead of 404
        return False
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        with z.open(name) as f:
            df = pd.read_csv(f, usecols=lambda c: c in KEEP, low_memory=False)
    df = df[df["Reporting_Airline"] == "WN"]
    df.to_csv(out, index=False)
    print(f"   {len(df):,} Southwest flights")
    return True


def main(start: str, end: str):
    RAW.mkdir(parents=True, exist_ok=True)
    got = []
    for y, m in month_range(start, end):
        if not fetch_month(y, m):
            print(f"{y}-{m:02d}: not published yet. Stopping here (latest available is the month before).")
            break
        got.append((y, m))
    if not got:
        raise SystemExit("No months downloaded. Check --start/--end.")
    files = [RAW / f"wn_{y}_{m:02d}.csv" for y, m in got]
    df = pd.concat((pd.read_csv(f, low_memory=False) for f in files), ignore_index=True)
    out = ROOT / "data" / "southwest_flights.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df):,} rows ({got[0][0]}-{got[0][1]:02d} to {got[-1][0]}-{got[-1][1]:02d}) to {out}")


if __name__ == "__main__":
    today = pd.Period(date.today(), "M")
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=str(today - 24), help="YYYY-MM (default: 24 months ago)")
    ap.add_argument("--end", default=str(today - 1), help="YYYY-MM (default: last month)")
    a = ap.parse_args()
    main(a.start, a.end)
