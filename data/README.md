# Data

Run `python src/download_data.py` (last 24 months up to the latest published; use `--start YYYY-MM` to change). Months are cached in `data/raw/`, so re-running resumes.
It downloads BTS "Reporting Carrier On-Time Performance" monthly files from
https://transtats.bts.gov/PREZIP/ and keeps Southwest (`Reporting_Airline == "WN"`).

If the download fails, open https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ ,
select the columns listed in `src/download_data.py` (KEEP), download each month, filter to WN,
and save as `data/southwest_flights.csv`.

Public U.S. government data (no license restrictions).
