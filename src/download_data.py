"""
Download Bureau of Transportation Statistics (BTS) on-time performance data
and keep only Southwest Airlines (carrier code WN) flights.

Usage:
  python src/download_data.py --year 2024 --months 4 5 6
Output:
  data/southwest_flights.csv
"""
import argparse
import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
URL = ("https://transtats.bts.gov/PREZIP/"
       "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{y}_{m}.zip")
KEEP = [
    "FlightDate", "Month", "DayofMonth", "DayOfWeek", "Reporting_Airline",
    "Origin", "Dest", "CRSDepTime", "CRSArrTime", "Distance",
    "ArrDelay", "ArrDel15", "Cancelled", "Diverted",
    "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay",
]


def fetch_month(year: int, month: int) -> pd.DataFrame:
    url = URL.format(y=year, m=month)
    print(f"Downloading {year}-{month:02d} ...")
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        with z.open(name) as f:
            df = pd.read_csv(f, usecols=lambda c: c in KEEP, low_memory=False)
    df = df[df["Reporting_Airline"] == "WN"]
    print(f"  {len(df):,} Southwest flights")
    return df


def main(year: int, months: list[int]):
    out = ROOT / "data" / "southwest_flights.csv"
    df = pd.concat([fetch_month(year, m) for m in months], ignore_index=True)
    df.to_csv(out, index=False)
    print(f"Saved {len(df):,} rows to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--months", type=int, nargs="+", default=[4, 5, 6])
    a = ap.parse_args()
    main(a.year, a.months)
