"""
Streamlit dashboard: Southwest on-time performance + flight delay risk.
Run:  streamlit run app.py   (after python src/train.py)
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
st.set_page_config(page_title="TurnTime", layout="wide")


@st.cache_resource
def load():
    return (joblib.load(ROOT / "models" / "model.joblib"),
            json.loads((ROOT / "results" / "metrics.json").read_text()),
            pd.read_csv(ROOT / "results" / "airport_stats.csv", index_col=0),
            pd.read_csv(ROOT / "results" / "delay_by_hour.csv", index_col=0))


try:
    bundle, m, airports, by_hour = load()
except FileNotFoundError:
    st.error("Run `python src/train.py` first.")
    st.stop()

k, mdl = m["ops"]["kpis"], m["model"]
best = next(r for r in mdl["models"] if r["model"] == mdl["best_model"])

st.title("TurnTime: Southwest Delay Intelligence")
st.caption(f"{k['flights']:,} flights · {k['airports']} airports · BTS on-time performance data")

c = st.columns(4)
c[0].metric("On-time arrival rate", f"{k['on_time_rate']:.1%}")
c[1].metric("Cancellation rate", f"{k['cancel_rate']:.2%}")
c[2].metric("Avg delay when late", f"{k['avg_delay_min_when_late']:.0f} min")
c[3].metric("Delays caught in top-20% risk", f"{best['capture_top20_pct']:.0f}%")

tab1, tab2 = st.tabs(["Operations", "Delay risk predictor"])

with tab1:
    a, b = st.columns(2)
    a.subheader("Delay minutes by cause")
    a.bar_chart(pd.Series(m["ops"]["cause_share_pct"]))
    b.subheader("Delay rate by departure hour")
    b.line_chart(by_hour["delay_rate"] * 100)
    st.subheader("Airport delay rates")
    big = airports[airports["flights"] >= airports["flights"].quantile(0.5)]
    st.dataframe((big.assign(delay_rate_pct=lambda x: (x["rate"] * 100).round(1))
                  .drop(columns="rate").sort_values("delay_rate_pct", ascending=False)),
                 width="stretch")

with tab2:
    L = bundle["lookups"]
    routes = sorted(bundle["routes"])
    route = st.selectbox("Route", routes,
                         index=routes.index("DAL-HOU") if "DAL-HOU" in routes else 0)
    origin, dest = route.split("-")
    hour = st.slider("Scheduled departure hour", 5, 23, 17)
    dow = st.selectbox("Day of week", list(range(1, 8)), index=4,
                       format_func=lambda d: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][d - 1])
    prior = bundle["prior"]
    row = pd.DataFrame([{
        "dep_hour": hour, "arr_hour": min(hour + 2, 23), "day_of_week": dow,
        "is_weekend": int(dow >= 6), "distance": bundle["routes"][route],
        "origin_hour_departures": bundle["median_congestion"],
        "route_hist_delay": L["route_hist_delay"].get(route, prior),
        "origin_hist_delay": L["origin_hist_delay"].get(origin, prior),
        "dest_hist_delay": L["dest_hist_delay"].get(dest, prior),
        "hour_hist_delay": L["hour_hist_delay"].get(hour, prior),
        "Origin": origin, "Dest": dest,
    }])[bundle["features"]]
    p = float(bundle["pipeline"].predict_proba(row)[:, 1][0])
    st.metric("Delay risk score", f"{p:.0%}")
    st.progress(min(p, 1.0))
    st.caption("Model trained with class balancing, so scores rank risk rather than give exact probabilities. "
               "Use them to prioritize turnaround staffing and spare-aircraft planning.")

st.divider()
f1, f2 = st.columns(2)
figs = ROOT / "results" / "figures"
f1.image(str(figs / "capture_curve.png"))
f2.image(str(figs / "feature_importance.png"))
