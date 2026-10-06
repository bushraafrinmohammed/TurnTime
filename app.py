"""
TurnTime: interactive Southwest on-time performance and delay-risk dashboard.
Run:  python -m streamlit run app.py   (after src/run_sql.py and src/train.py)
"""
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent
RES = ROOT / "results"
NAVY, ORANGE, GOLD, SKY, GREY = "#13294B", "#E4572E", "#FFB347", "#4A90D9", "#94A3B8"
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
CAUSE_LABELS = {"LateAircraftDelay": "Late-arriving aircraft", "CarrierDelay": "Airline (carrier)",
                "NASDelay": "Air traffic / airspace", "WeatherDelay": "Weather", "SecurityDelay": "Security"}

st.set_page_config(page_title="TurnTime", page_icon="✈️", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(ROOT / "models" / "model.joblib")


@st.cache_data
def load_data():
    d = RES / "dashboard"
    return {
        "m": json.loads((RES / "metrics.json").read_text()),
        "cube": pd.read_csv(d / "cube.csv.gz").assign(
            date_ts=lambda x: pd.to_datetime(x["date"]), dow=lambda x: pd.to_datetime(x["date"]).dt.dayofweek + 1),
        "routes_month": pd.read_csv(d / "routes_month.csv.gz"),
    }


try:
    bundle, D = load_model(), load_data()
except FileNotFoundError:
    st.error("Results not found. Run `python src/run_sql.py` and `python src/train.py` first.")
    st.stop()

m, k, mdl = D["m"], D["m"]["ops"]["kpis"], D["m"]["model"]
best = next(r for r in mdl["models"] if r["model"] == mdl["best_model"])
eve = m["ops"]["morning_vs_evening"]

# ---------------- Style ----------------
st.markdown(f"""
<style>
.hero {{background: linear-gradient(120deg, {NAVY} 0%, #2E4A7D 70%, {ORANGE} 140%); border-radius: 18px;
        padding: 28px 32px; margin-bottom: 18px; color: #fff;}}
.hero h1 {{margin: 0; font-size: 2.4rem; letter-spacing: -0.5px; color: #fff;}}
.hero .tag {{font-size: 1.05rem; opacity: 0.92; margin-top: 4px;}}
.hero .sub {{font-size: 0.85rem; opacity: 0.75; margin-top: 10px;}}
.chip {{display: inline-block; background: rgba(255,255,255,0.14); border: 1px solid rgba(255,255,255,0.25);
        border-radius: 999px; padding: 3px 12px; margin: 10px 6px 0 0; font-size: 0.8rem;}}
.bar {{width: 56px; height: 4px; background: {GOLD}; border-radius: 2px; margin-bottom: 12px;}}
div[data-testid="stMetric"] {{background: #fff; border: 1px solid #D9E1EC; border-radius: 14px;
        padding: 14px 16px; box-shadow: 0 1px 3px rgba(19,41,75,0.06);}}
.insight {{background: #FFF7EC; border-left: 4px solid {ORANGE}; border-radius: 10px;
        padding: 12px 16px; margin: 6px 0 14px 0; color: {NAVY};}}
.verdict {{border-radius: 12px; padding: 14px 18px; font-weight: 600; margin-top: 8px;}}
</style>""", unsafe_allow_html=True)

_months = f"{mdl['train_months'][0]} to {mdl['test_month']}"
st.markdown(f"""<div class="hero"><div class="bar"></div><h1>✈️ TurnTime</h1>
<div class="tag">Southwest Airlines on-time performance &amp; delay risk intelligence</div>
<div><span class="chip">SQL window functions</span><span class="chip">Time-based validation</span>
<span class="chip">Leakage-free features</span><span class="chip">{k['flights']:,} flights</span></div>
<div class="sub">U.S. DOT on-time data · {k['airports']} airports · {k['routes']:,} routes · {_months}</div></div>""",
            unsafe_allow_html=True)


def insight(text):
    st.markdown(f'<div class="insight">💡 {text}</div>', unsafe_allow_html=True)


def style(fig, h=380):
    fig.update_layout(template="plotly_white", height=h, margin=dict(l=10, r=10, t=40, b=10),
                      font=dict(color=NAVY), legend=dict(orientation="h", y=-0.2))
    return fig


# ---------------- Sidebar (global filters) ----------------
cube = D["cube"]
dates = sorted(cube["date"].unique())
with st.sidebar:
    st.markdown("### 🎛️ Filters")
    st.caption("Every KPI and chart updates with these.")
    d0, d1 = pd.Timestamp(dates[0]).date(), pd.Timestamp(dates[-1]).date()
    dr = st.slider("Date range", d0, d1, (d0, d1), format="MMM D, YYYY")
    all_ap = cube.groupby("Origin")["flights"].sum().sort_values(ascending=False).index.tolist()
    sel_ap = st.multiselect("Airports (departing)", all_ap, placeholder="All airports",
                            help="Leave empty for all airports")
    sel_days = st.multiselect("Days of week", DAYS, placeholder="All days")
    hr_rng = st.slider("Departure hours", 0, 23, (0, 23))
    st.divider()
    st.markdown("### 🛫 Airport ranking")
    min_flights = st.slider("Minimum flights per airport", 50, 5000, 500, 50,
                            help="Hides small airports so rankings aren't driven by noise")
    top_n = st.slider("Airports to show", 5, 30, 15)
    metric_choice = st.radio("Rank airports by", ["Delay rate", "Avg delay (min)", "Cancellation rate"])
    st.divider()
    st.markdown("### 📌 About")
    st.caption("Built with Python, SQL (DuckDB), scikit-learn and Streamlit on public U.S. DOT data. "
               "The delay model uses only information known before departure.")

f = cube[(cube["date_ts"] >= pd.Timestamp(dr[0])) & (cube["date_ts"] <= pd.Timestamp(dr[1]))
         & cube["dep_hour"].between(*hr_rng)]
if sel_ap:
    f = f[f["Origin"].isin(sel_ap)]
if sel_days:
    f = f[f["dow"].isin([DAYS.index(d) + 1 for d in sel_days])]
filtered = len(sel_ap) + len(sel_days) > 0 or hr_rng != (0, 23) or dr != (d0, d1)


def kpis(x):
    fl, fn, lt = x["flights"].sum(), x["flown"].sum(), x["late"].sum()
    return dict(flights=int(fl), on_time=1 - lt / fn if fn else np.nan,
                cancel=x["cancelled"].sum() / fl if fl else np.nan,
                avg_delay=x["late_delay_min"].sum() / lt if lt else np.nan)


K, KA = kpis(f), kpis(cube)
if f.empty:
    st.warning("No flights match these filters. Widen them in the sidebar.")
    st.stop()

c = st.columns(5)
c[0].metric("Flights" + (" (filtered)" if filtered else ""), f"{K['flights']:,}",
            f"{K['flights'] / KA['flights']:.0%} of all" if filtered else None, delta_color="off")
c[1].metric("On-time arrival", f"{K['on_time']:.1%}",
            f"{100 * (K['on_time'] - KA['on_time']):+.1f} pts vs all" if filtered else None)
c[2].metric("Cancellation rate", f"{K['cancel']:.2%}",
            f"{100 * (K['cancel'] - KA['cancel']):+.2f} pts vs all" if filtered else None, delta_color="inverse")
c[3].metric("Avg delay when late", f"{K['avg_delay']:.0f} min",
            f"{K['avg_delay'] - KA['avg_delay']:+.0f} min vs all" if filtered else None, delta_color="inverse")
c[4].metric("Model: delays caught in top 20%", f"{best['capture_top20_pct']:.0f}%",
            f"{best['capture_top20_pct'] / 20:.1f}x vs random")

tabs = st.tabs(["📊 Overview", "🛫 Airports", "🧭 Routes", "🔥 Heatmap", "🧠 Delay Predictor",
                "🧮 SQL Explorer", "📈 Model"])

# ---------------- Overview ----------------
with tabs[0]:
    hr = f.groupby("dep_hour")[["flights", "flown", "late"]].sum().reset_index()
    hr["delay_rate"] = hr["late"] / hr["flown"].where(hr["flown"] > 0)
    am = hr.loc[hr["dep_hour"] < 9, "late"].sum() / max(hr.loc[hr["dep_hour"] < 9, "flown"].sum(), 1)
    pm = hr.loc[hr["dep_hour"] >= 17, "late"].sum() / max(hr.loc[hr["dep_hour"] >= 17, "flown"].sum(), 1)
    causes = f[list(CAUSE_LABELS)].sum()
    top_cause = causes.idxmax()
    share = 100 * causes[top_cause] / max(causes.sum(), 1)
    scope = "in this selection" if filtered else "overall"
    insight(f"<b>Delays snowball through the day</b> {scope}: {am:.1%} of flights before 9 AM arrive late vs "
            f"<b>{pm:.1%}</b> after 5 PM. The #1 cause is <b>{CAUSE_LABELS[top_cause].lower()}</b> "
            f"({share:.1f}% of delay minutes).")
    a, b = st.columns([3, 2])
    daily = f.groupby("date_ts")[["flown", "late"]].sum().reset_index().sort_values("date_ts")
    daily["on_time"] = 100 * (1 - daily["late"] / daily["flown"].where(daily["flown"] > 0))
    daily["rolling"] = daily["on_time"].rolling(7, min_periods=1).mean()
    fig = go.Figure()
    fig.add_bar(x=daily["date_ts"], y=daily["on_time"], name="Daily on-time %", marker_color="#C9D6EA",
                hovertemplate="%{x|%b %d}: %{y:.1f}%<extra></extra>")
    fig.add_scatter(x=daily["date_ts"], y=daily["rolling"], name="7-day average", line=dict(color=ORANGE, width=3))
    fig.update_yaxes(title="On-time %", range=[max(0, daily["on_time"].min() - 10), 100])
    a.plotly_chart(style(fig).update_layout(title="Daily on-time performance"), width="stretch")

    cs = causes.rename(index=CAUSE_LABELS)
    donut = go.Figure(go.Pie(labels=cs.index, values=cs.values, hole=0.6, sort=False,
                             marker=dict(colors=[ORANGE, NAVY, SKY, GOLD, GREY])))
    donut.update_traces(textinfo="percent", hovertemplate="%{label}: %{value:,.0f} min<extra></extra>")
    b.plotly_chart(style(donut).update_layout(title="What causes delay minutes?"), width="stretch")

    fig = go.Figure()
    fig.add_bar(x=hr["dep_hour"], y=hr["flights"], name="Flights", marker_color="#DCE5F2", yaxis="y2",
                hovertemplate="%{x}:00 · %{y:,} flights<extra></extra>")
    fig.add_scatter(x=hr["dep_hour"], y=100 * hr["delay_rate"], name="% arriving late", mode="lines+markers",
                    line=dict(color=ORANGE, width=3), hovertemplate="%{x}:00 · %{y:.1f}% late<extra></extra>")
    fig.update_layout(yaxis=dict(title="% arriving 15+ min late"),
                      yaxis2=dict(overlaying="y", side="right", showgrid=False, title="Flights"),
                      xaxis=dict(title="Scheduled departure hour", dtick=2))
    st.plotly_chart(style(fig).update_layout(title="Delay propagation by departure hour"), width="stretch")

    cm = f.assign(month=f["date"].str[:7]).groupby("month")[list(CAUSE_LABELS)].sum().rename(columns=CAUSE_LABELS)
    cm = (cm.div(cm.sum(axis=1).replace(0, np.nan), axis=0) * 100).reset_index().melt(
        "month", var_name="cause", value_name="pct")
    fig = px.bar(cm, x="month", y="pct", color="cause", barmode="stack",
                 color_discrete_sequence=[ORANGE, NAVY, SKY, GOLD, GREY])
    fig.update_yaxes(title="% of delay minutes")
    st.plotly_chart(style(fig, 340).update_layout(title="Delay-cause mix by month"), width="stretch")

# ---------------- Airports ----------------
with tabs[1]:
    ap = f.groupby("Origin")[["flights", "cancelled", "flown", "late", "late_delay_min"]].sum().reset_index()
    ap = ap[ap["flights"] >= min_flights]
    ap["Delay rate"] = 100 * ap["late"] / ap["flown"].where(ap["flown"] > 0)
    ap["Avg delay (min)"] = ap["late_delay_min"] / ap["late"].where(ap["late"] > 0)
    ap["Cancellation rate"] = 100 * ap["cancelled"] / ap["flights"]
    if ap.empty:
        st.warning("No airports meet the minimum-flights filter for this selection. Lower it in the sidebar.")
    else:
        show = ap.sort_values(metric_choice, ascending=False).head(top_n)
        unit = " min" if "min" in metric_choice else "%"
        insight(f"Among {len(ap)} airports with {min_flights:,}+ flights {scope}, <b>{show.iloc[0]['Origin']}</b> "
                f"ranks worst on {metric_choice.lower()} ({show.iloc[0][metric_choice]:.1f}{unit}).")
        a, b = st.columns(2)
        fig = px.bar(show.sort_values(metric_choice), x=metric_choice, y="Origin", orientation="h",
                     color=metric_choice, color_continuous_scale=["#C9D6EA", ORANGE], hover_data={"flights": ":,"})
        a.plotly_chart(style(fig, 520).update_layout(title=f"Top {len(show)} airports by {metric_choice.lower()}",
                                                     coloraxis_showscale=False), width="stretch")
        fig = px.scatter(ap, x="flights", y="Delay rate", size="flights", color="Avg delay (min)",
                         hover_name="Origin", color_continuous_scale=[SKY, ORANGE], log_x=True, size_max=40)
        fig.update_xaxes(title="Flights (log scale)")
        fig.update_yaxes(title="% arriving late")
        b.plotly_chart(style(fig, 520).update_layout(title="Volume vs delay rate (hover any bubble)"),
                       width="stretch")
        st.dataframe(ap[["Origin", "flights", "Delay rate", "Avg delay (min)", "Cancellation rate"]]
                     .sort_values(metric_choice, ascending=False).round(1), width="stretch", hide_index=True)

# ---------------- Routes ----------------
with tabs[2]:
    months = {pd.Timestamp(d).strftime("%Y-%m") for d in pd.date_range(dr[0], dr[1], freq="D")}
    rt = D["routes_month"][D["routes_month"]["month"].isin(months)]
    rt = (rt.groupby(["Origin", "Dest"]).agg(flights=("flights", "sum"), late=("late", "sum"),
                                              late_delay_min=("late_delay_min", "sum"), distance=("distance", "median"))
          .reset_index().query("flights >= 20"))
    rt["route"] = rt["Origin"] + " → " + rt["Dest"]
    rt["Delay rate"] = 100 * rt["late"] / rt["flights"]
    st.caption(f"Routes use whole months overlapping your date range ({', '.join(sorted(months))}).")
    origins = sorted(rt["Origin"].unique())
    pref = sel_ap[0] if sel_ap and sel_ap[0] in origins else "DAL"
    a, b = st.columns([1, 3])
    org = a.selectbox("Departing from", origins, index=origins.index(pref) if pref in origins else 0)
    sort_by = a.radio("Sort destinations by", ["Delay rate", "Flights"])
    sub = rt[rt["Origin"] == org].sort_values("Delay rate" if sort_by == "Delay rate" else "flights",
                                               ascending=False).head(25)
    a.metric("Destinations served", int((rt["Origin"] == org).sum()))
    a.metric("Flights from here", f"{int(rt.loc[rt['Origin'] == org, 'flights'].sum()):,}")
    fig = px.bar(sub, x="Dest", y="Delay rate", color="Delay rate", color_continuous_scale=["#C9D6EA", ORANGE],
                 hover_data={"flights": ":,", "distance": ":.0f"})
    fig.update_yaxes(title="% arriving late")
    b.plotly_chart(style(fig, 420).update_layout(title=f"Routes from {org}", coloraxis_showscale=False),
                   width="stretch")
    st.markdown("#### 🚨 Routes losing the most total time")
    worst = rt.assign(total_delay_hrs=rt["late_delay_min"] / 60).sort_values("total_delay_hrs", ascending=False).head(10)
    fig = px.bar(worst.sort_values("total_delay_hrs"), x="total_delay_hrs", y="route", orientation="h",
                 color_discrete_sequence=[NAVY], hover_data={"flights": ":,", "Delay rate": ":.1f"})
    fig.update_xaxes(title="Total hours of arrival delay")
    st.plotly_chart(style(fig, 380), width="stretch")
    insight("High-volume routes with moderate delay rates often lose more total hours than small, very "
            "late routes. That's where turnaround improvements pay off most.")

# ---------------- Heatmap ----------------
with tabs[3]:
    h = f.groupby(["dow", "dep_hour"])[["flown", "late"]].sum()
    h = h[h["flown"] >= 20]
    if h.empty:
        st.info("Not enough flights in this selection for a heatmap.")
    else:
        hm = (100 * h["late"] / h["flown"]).unstack("dep_hour")
        hm.index = [DAYS[i - 1] for i in hm.index]
        fig = px.imshow(hm, aspect="auto", color_continuous_scale=["#F1F5FB", GOLD, ORANGE, "#8B1E1E"],
                        labels=dict(x="Departure hour", y="", color="% late"), text_auto=".0f")
        st.plotly_chart(style(fig, 420).update_layout(title="When are delays worst? (% of flights arriving late)"),
                        width="stretch")
        st_ = hm.stack()
        insight(f"Worst slot {scope}: <b>{st_.idxmax()[0]} {st_.idxmax()[1]}:00</b> ({st_.max():.0f}% late). "
                f"Best slot: <b>{st_.idxmin()[0]} {st_.idxmin()[1]}:00</b> ({st_.min():.0f}% late).")

# ---------------- Predictor ----------------
with tabs[4]:
    L, prior = bundle["lookups"], bundle["prior"]
    routes = sorted(bundle["routes"])
    a, b = st.columns([1, 2])
    with a:
        route = st.selectbox("Route", routes, index=routes.index("DAL-HOU") if "DAL-HOU" in routes else 0)
        dow = st.select_slider("Day of week", options=list(range(1, 8)), value=5,
                               format_func=lambda d: DAYS[d - 1])
        hour = st.slider("Scheduled departure hour", 5, 23, 17)
    origin, dest = route.split("-")

    def frame(hours):
        return pd.DataFrame([{
            "dep_hour": h, "arr_hour": min(h + 2, 23), "day_of_week": dow, "is_weekend": int(dow >= 6),
            "distance": bundle["routes"][route], "origin_hour_departures": bundle["median_congestion"],
            "route_hist_delay": L["route_hist_delay"].get(route, prior),
            "origin_hist_delay": L["origin_hist_delay"].get(origin, prior),
            "dest_hist_delay": L["dest_hist_delay"].get(dest, prior),
            "hour_hist_delay": L["hour_hist_delay"].get(h, prior),
            "Origin": origin, "Dest": dest} for h in hours])[bundle["features"]]

    hours = list(range(5, 24))
    scores = bundle["pipeline"].predict_proba(frame(hours))[:, 1]
    p = float(scores[hours.index(hour)])
    gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=100 * p, number={"suffix": "", "font": {"size": 44}},
        title={"text": f"Risk score · {route.replace('-', ' → ')}"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": NAVY},
               "steps": [{"range": [0, 40], "color": "#DFF3E4"}, {"range": [40, 60], "color": "#FFF1D6"},
                         {"range": [60, 100], "color": "#FBD9D0"}]}))
    with a:
        st.plotly_chart(style(gauge, 260).update_layout(margin=dict(t=60, b=0)), width="stretch")
        level = ("🟢 Low risk", "#DFF3E4") if p < 0.4 else ("🟡 Moderate risk", "#FFF1D6") if p < 0.6 \
            else ("🔴 High risk", "#FBD9D0")
        st.markdown(f'<div class="verdict" style="background:{level[1]}">{level[0]}</div>', unsafe_allow_html=True)
    best_h = hours[int(np.argmin(scores))]
    fig = go.Figure()
    fig.add_scatter(x=hours, y=100 * scores, mode="lines+markers", line=dict(color=ORANGE, width=3),
                    hovertemplate="%{x}:00 · score %{y:.0f}<extra></extra>", name="Risk score")
    fig.add_vline(x=hour, line_dash="dash", line_color=NAVY, annotation_text="Selected")
    fig.add_vline(x=best_h, line_dash="dot", line_color="#2E9E5B", annotation_text="Lowest risk")
    fig.update_xaxes(title="Departure hour", dtick=1)
    fig.update_yaxes(title="Risk score (0-100)")
    b.plotly_chart(style(fig, 420).update_layout(title=f"Best time to depart on {DAYS[dow - 1]}"),
                   width="stretch")
    b.caption("Scores rank relative risk (model trained with class balancing). Use them to prioritize "
              "turnaround staffing and spare aircraft, not as exact probabilities.")
    insight(f"For {route.replace('-', ' → ')} on {DAYS[dow - 1]}, departing at <b>{best_h}:00</b> carries the "
            f"lowest risk score ({100 * scores.min():.0f}) vs {100 * p:.0f} at {hour}:00.")

# ---------------- SQL Explorer ----------------
with tabs[5]:
    sql_path, sql_dir = ROOT / "sql" / "kpi_queries.sql", RES / "sql"
    blocks = re.split(r"^-- name:\s*(\S+)\s*$", sql_path.read_text(), flags=re.M)[1:]
    queries = dict(zip(blocks[::2], blocks[1::2]))
    if not sql_dir.exists():
        st.info("Run `python src/run_sql.py` to generate SQL results.")
    else:
        name = st.selectbox("Choose a query", list(queries),
                            format_func=lambda n: n.split("_", 1)[1].replace("_", " ").title())
        a, b = st.columns([1, 1])
        a.markdown("**SQL (DuckDB)**")
        a.code(queries[name].strip(), language="sql")
        res = pd.read_csv(sql_dir / f"{name}.csv")
        b.markdown(f"**Result** · {len(res)} rows")
        b.dataframe(res, width="stretch", hide_index=True, height=420)
        num = res.select_dtypes("number").columns
        if len(res.columns) >= 2 and len(num):
            ycol = num[-1] if res.columns[0] in num else num[0]
            fig = px.bar(res, x=res.columns[0], y=ycol, color_discrete_sequence=[NAVY])
            st.plotly_chart(style(fig, 320).update_layout(title=f"{ycol} by {res.columns[0]}"), width="stretch")

# ---------------- Model ----------------
with tabs[6]:
    res = pd.DataFrame(mdl["models"])
    melt = res.melt("model", ["roc_auc", "pr_auc"], var_name="metric", value_name="score")
    a, b = st.columns(2)
    fig = px.bar(melt, x="metric", y="score", color="model", barmode="group",
                 color_discrete_sequence=[NAVY, ORANGE], text_auto=".3f")
    a.plotly_chart(style(fig, 360).update_layout(title=f"Test month {mdl['test_month']}"), width="stretch")
    imp = pd.Series(mdl["top_features"]).sort_values()
    fig = px.bar(x=imp.values, y=imp.index, orientation="h", color_discrete_sequence=[ORANGE])
    fig.update_xaxes(title="Drop in ROC-AUC when shuffled")
    fig.update_yaxes(title="")
    b.plotly_chart(style(fig, 360).update_layout(title="What drives delay risk?"), width="stretch")
    st.image(str(RES / "figures" / "capture_curve.png"), width=640)
    insight(f"Trained on {', '.join(mdl['train_months'])} and tested on the unseen month {mdl['test_month']}. "
            f"The simpler model is kept when extra complexity doesn't improve results.")
