-- Southwest Airlines On-Time Performance: KPI queries (DuckDB SQL; standard SQL with window functions)
-- Table: flights (loaded from data/southwest_flights.csv by src/run_sql.py)
-- Each query is separated by a "-- name:" header so the runner can save results individually.

-- name: 01_monthly_kpis
-- Monthly on-time %, cancellation %, average delay when late, and month-over-month change
WITH monthly AS (
    SELECT
        date_trunc('month', CAST(FlightDate AS DATE))                 AS month,
        COUNT(*)                                                      AS flights,
        AVG(CASE WHEN Cancelled = 0 AND Diverted = 0 THEN 1 - ArrDel15 END) AS on_time_rate,
        AVG(Cancelled)                                                AS cancel_rate,
        AVG(CASE WHEN ArrDel15 = 1 THEN ArrDelay END)                 AS avg_delay_when_late
    FROM flights
    GROUP BY 1
)
SELECT
    month,
    flights,
    ROUND(100 * on_time_rate, 1)                                      AS on_time_pct,
    ROUND(100 * cancel_rate, 2)                                       AS cancel_pct,
    ROUND(avg_delay_when_late, 1)                                     AS avg_delay_min,
    ROUND(100 * (on_time_rate - LAG(on_time_rate) OVER (ORDER BY month)), 1) AS on_time_change_pts
FROM monthly
ORDER BY month;

-- name: 02_delay_cause_minutes
-- Share of total delay minutes by cause (unpivoted)
WITH causes AS (
    UNPIVOT (
        SELECT CarrierDelay, LateAircraftDelay, NASDelay, WeatherDelay, SecurityDelay
        FROM flights WHERE ArrDel15 = 1
    ) ON CarrierDelay, LateAircraftDelay, NASDelay, WeatherDelay, SecurityDelay
    INTO NAME cause VALUE minutes
)
SELECT
    cause,
    ROUND(SUM(minutes))                                               AS total_minutes,
    ROUND(100 * SUM(minutes) / SUM(SUM(minutes)) OVER (), 1)          AS pct_of_delay_minutes
FROM causes
GROUP BY cause
ORDER BY total_minutes DESC;

-- name: 03_airport_ranking
-- Origin airports ranked by delay rate (min 500 flights), with rank and percentile
SELECT
    Origin                                                            AS airport,
    COUNT(*)                                                          AS flights,
    ROUND(100 * AVG(ArrDel15), 1)                                     AS delay_pct,
    RANK() OVER (ORDER BY AVG(ArrDel15) DESC)                         AS delay_rank,
    ROUND(100 * PERCENT_RANK() OVER (ORDER BY AVG(ArrDel15)), 0)      AS percentile
FROM flights
WHERE Cancelled = 0 AND Diverted = 0
GROUP BY Origin
HAVING COUNT(*) >= 500
ORDER BY delay_pct DESC
LIMIT 15;

-- name: 04_worst_routes
-- Routes with the most total delay minutes (where operational fixes pay off most)
SELECT
    Origin || '-' || Dest                                             AS route,
    COUNT(*)                                                          AS flights,
    ROUND(100 * AVG(ArrDel15), 1)                                     AS delay_pct,
    ROUND(SUM(CASE WHEN ArrDel15 = 1 THEN ArrDelay ELSE 0 END))       AS total_delay_minutes
FROM flights
WHERE Cancelled = 0 AND Diverted = 0
GROUP BY 1
HAVING COUNT(*) >= 100
ORDER BY total_delay_minutes DESC
LIMIT 15;

-- name: 05_delay_by_hour
-- Delay propagation through the day, with a running cumulative share of all delays
WITH hourly AS (
    SELECT
        CAST(CRSDepTime / 100 AS INTEGER)                             AS dep_hour,
        COUNT(*)                                                      AS flights,
        SUM(ArrDel15)                                                 AS delayed
    FROM flights
    WHERE Cancelled = 0 AND Diverted = 0
    GROUP BY 1
)
SELECT
    dep_hour,
    flights,
    ROUND(100.0 * delayed / flights, 1)                               AS delay_pct,
    ROUND(100.0 * SUM(delayed) OVER (ORDER BY dep_hour) / SUM(delayed) OVER (), 1) AS cumulative_pct_of_delays
FROM hourly
ORDER BY dep_hour;

-- name: 06_day_of_week
-- Day-of-week pattern vs. overall average
SELECT
    DayOfWeek                                                         AS day_of_week,
    COUNT(*)                                                          AS flights,
    ROUND(100 * AVG(ArrDel15), 1)                                     AS delay_pct,
    ROUND(100 * (AVG(ArrDel15) - AVG(AVG(ArrDel15)) OVER ()), 1)      AS vs_avg_pts
FROM flights
WHERE Cancelled = 0 AND Diverted = 0
GROUP BY DayOfWeek
ORDER BY DayOfWeek;
