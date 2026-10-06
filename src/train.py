"""
Southwest Airlines: Flight Delay Analytics and Prediction

Part A, Operations analytics:
  - On-time performance KPIs
  - Delay-minute breakdown by cause (carrier, late aircraft, NAS, weather, security)
  - Delay propagation through the day (delay rate by scheduled departure hour)
  - Airports with the worst arrival delay rates

Part B, Delay prediction (arrival delay >= 15 min) using ONLY information known
before departure (schedule, route, day, airport congestion). No DepDelay, no
delay-cause columns: those are known only after the flight and would leak the target.
  - Time-based split: last month = test (mimics predicting the future)
  - Route/airport historical delay rates computed on TRAIN only
  - Baseline Logistic Regression vs HistGradientBoosting
  - Business metric: share of delayed flights captured in the top 20% riskiest

Usage:
  python src/train.py --data data/southwest_flights.csv
"""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
RESULTS, FIGS, MODELS = ROOT / "results", ROOT / "results" / "figures", ROOT / "models"
CAUSES = ["CarrierDelay", "LateAircraftDelay", "NASDelay", "WeatherDelay", "SecurityDelay"]
NUM = ["dep_hour", "arr_hour", "day_of_week", "is_weekend", "distance",
       "origin_hour_departures", "route_hist_delay", "origin_hist_delay", "dest_hist_delay",
       "hour_hist_delay"]
CAT = ["Origin", "Dest"]


# ---------------- Data ----------------
def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["FlightDate"] = pd.to_datetime(df["FlightDate"])
    df["dep_hour"] = (df["CRSDepTime"] // 100).clip(0, 23).astype(int)
    df["arr_hour"] = (df["CRSArrTime"] // 100).clip(0, 23).astype(int)
    df["day_of_week"] = df["DayOfWeek"].astype(int)
    df["is_weekend"] = df["day_of_week"].isin([6, 7]).astype(int)
    df["distance"] = df["Distance"].astype(float)
    df["route"] = df["Origin"] + "-" + df["Dest"]
    # Scheduled congestion: departures from same airport, same day, same hour (known in advance)
    df["origin_hour_departures"] = df.groupby(["FlightDate", "Origin", "dep_hour"])["Origin"].transform("size")
    df["ym"] = df["FlightDate"].dt.to_period("M")
    return df


# ---------------- Part A ----------------
def ops_analytics(df: pd.DataFrame) -> dict:
    flown = df[(df["Cancelled"] == 0) & (df["Diverted"] == 0)]
    cause_min = flown[CAUSES].fillna(0).sum()
    cause_share = (cause_min / cause_min.sum() * 100).sort_values(ascending=False)

    by_hour = flown.groupby("dep_hour")["ArrDel15"].agg(["mean", "size"])
    by_hour = by_hour[by_hour["size"] >= 200]

    ap = flown.groupby("Origin")["ArrDel15"].agg(rate="mean", flights="size")
    min_flights = max(300, int(len(flown) * 0.005))
    worst = ap[ap["flights"] >= min_flights].sort_values("rate", ascending=False).head(10)

    kpis = {
        "flights": int(len(df)),
        "on_time_rate": float(1 - flown["ArrDel15"].mean()),
        "cancel_rate": float(df["Cancelled"].mean()),
        "avg_delay_min_when_late": float(flown.loc[flown["ArrDel15"] == 1, "ArrDelay"].mean()),
        "airports": int(df["Origin"].nunique()),
        "routes": int(df["route"].nunique()),
    }

    # Figures
    fig, ax = plt.subplots(figsize=(6, 3.5))
    cause_share[::-1].plot.barh(ax=ax, color="#3b6fb6")
    ax.set_xlabel("% of total delay minutes"); ax.set_title("What causes Southwest delays?")
    fig.tight_layout(); fig.savefig(FIGS / "delay_causes.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(by_hour.index, by_hour["mean"] * 100, marker="o", color="#d9822b")
    ax.set_xlabel("Scheduled departure hour"); ax.set_ylabel("% flights arriving 15+ min late")
    ax.set_title("Delays build up through the day")
    fig.tight_layout(); fig.savefig(FIGS / "delay_by_hour.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    (worst["rate"][::-1] * 100).plot.barh(ax=ax, color="#b33a3a")
    ax.set_xlabel("% flights arriving 15+ min late"); ax.set_title("Highest-delay origin airports")
    fig.tight_layout(); fig.savefig(FIGS / "worst_airports.png", dpi=150); plt.close(fig)

    by_hour.rename(columns={"mean": "delay_rate", "size": "flights"}).to_csv(RESULTS / "delay_by_hour.csv")
    ap.sort_values("flights", ascending=False).to_csv(RESULTS / "airport_stats.csv")
    return {"kpis": kpis, "cause_share_pct": cause_share.round(1).to_dict(),
            "worst_airports": (worst["rate"] * 100).round(1).to_dict(),
            "peak_delay_hour": int(by_hour["mean"].idxmax()),
            "morning_vs_evening": {
                "before_9am": float(by_hour.loc[by_hour.index < 9, "mean"].mean()),
                "after_5pm": float(by_hour.loc[by_hour.index >= 17, "mean"].mean())}}


# ---------------- Part B ----------------
def add_history(train: pd.DataFrame, other: pd.DataFrame, prior: float, k: int = 50) -> pd.DataFrame:
    """Smoothed historical delay rates learned from TRAIN only (target encoding)."""
    other = other.copy()
    for key, col in [("route", "route_hist_delay"), ("Origin", "origin_hist_delay"),
                     ("Dest", "dest_hist_delay"), ("dep_hour", "hour_hist_delay")]:
        g = train.groupby(key)["ArrDel15"].agg(["sum", "count"])
        smooth = (g["sum"] + prior * k) / (g["count"] + k)
        other[col] = other[key].map(smooth).fillna(prior).astype(float)
    return other


def capture_at(y, p, frac=0.2):
    n = int(len(p) * frac)
    top = np.argsort(-p)[:n]
    return float(np.asarray(y)[top].sum() / np.asarray(y).sum())


def model_part(df: pd.DataFrame) -> dict:
    d = df[(df["Cancelled"] == 0) & (df["Diverted"] == 0)].dropna(subset=["ArrDel15"]).copy()
    d["ArrDel15"] = d["ArrDel15"].astype(int)
    last = d["ym"].max()
    train, test = d[d["ym"] < last], d[d["ym"] == last]
    prior = train["ArrDel15"].mean()

    # Out-of-fold style history for train: compute per train month from OTHER train months
    parts = []
    for m in train["ym"].unique():
        cur, rest = train[train["ym"] == m], train[train["ym"] != m]
        parts.append(add_history(rest if len(rest) else train, cur, prior))
    train_f = pd.concat(parts)
    test_f = add_history(train, test, prior)

    X_tr, y_tr = train_f[NUM + CAT], train_f["ArrDel15"]
    X_te, y_te = test_f[NUM + CAT], test_f["ArrDel15"]

    lr = Pipeline([
        ("pre", ColumnTransformer([
            ("num", StandardScaler(), NUM),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50), CAT)])),
        ("m", LogisticRegression(max_iter=2000, class_weight="balanced"))])

    hgb = Pipeline([
        ("pre", ColumnTransformer([
            ("num", "passthrough", NUM),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50, sparse_output=False), CAT)])),
        ("m", HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                             l2_regularization=1.0, class_weight="balanced",
                                             random_state=SEED))])

    rows, fitted = [], {}
    for name, pipe in [("Logistic Regression", lr), ("Gradient Boosting", hgb)]:
        pipe.fit(X_tr, y_tr)
        p = pipe.predict_proba(X_te)[:, 1]
        fitted[name] = (pipe, p)
        rows.append({"model": name, "roc_auc": roc_auc_score(y_te, p),
                     "pr_auc": average_precision_score(y_te, p),
                     "capture_top20_pct": 100 * capture_at(y_te, p)})
    res = pd.DataFrame(rows)
    best = res.sort_values("pr_auc", ascending=False).iloc[0]["model"]
    best_pipe, best_p = fitted[best]

    # Capture curve
    fig, ax = plt.subplots(figsize=(6, 4))
    fr = np.linspace(0.01, 1, 100)
    for name, (_, p) in fitted.items():
        ax.plot(fr * 100, [100 * capture_at(y_te, p, f) for f in fr], label=name)
    ax.plot(fr * 100, fr * 100, "--", c="gray", label="Random")
    ax.set_xlabel("% of flights flagged (highest risk first)")
    ax.set_ylabel("% of delayed flights captured"); ax.set_title(f"Delay capture curve, {last} test month")
    ax.legend(); fig.tight_layout(); fig.savefig(FIGS / "capture_curve.png", dpi=150); plt.close(fig)

    # Permutation importance (sample for speed)
    samp = X_te.sample(min(20000, len(X_te)), random_state=SEED)
    pi = permutation_importance(best_pipe, samp, y_te.loc[samp.index], scoring="roc_auc",
                                n_repeats=3, random_state=SEED, n_jobs=-1)
    imp = pd.Series(pi.importances_mean, index=samp.columns).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    imp.head(10)[::-1].plot.barh(ax=ax, color="#3b6fb6")
    ax.set_xlabel("Drop in ROC-AUC when shuffled"); ax.set_title("What predicts a delay?")
    fig.tight_layout(); fig.savefig(FIGS / "feature_importance.png", dpi=150); plt.close(fig)

    # Save model + lookup tables for the dashboard
    lookups = {c: test_f.groupby(k)[c].first().to_dict() for k, c in
               [("route", "route_hist_delay"), ("Origin", "origin_hist_delay"),
                ("Dest", "dest_hist_delay"), ("dep_hour", "hour_hist_delay")]}
    routes = d.groupby("route")["distance"].median().to_dict()
    joblib.dump({"pipeline": best_pipe, "features": NUM + CAT, "lookups": lookups,
                 "routes": routes, "prior": float(prior),
                 "median_congestion": float(train["origin_hour_departures"].median())},
                MODELS / "model.joblib")

    return {"train_months": [str(m) for m in sorted(train["ym"].unique())], "test_month": str(last),
            "train_rows": int(len(train)), "test_rows": int(len(test)),
            "test_delay_rate": float(y_te.mean()), "models": rows, "best_model": best,
            "top_features": imp.head(5).round(4).to_dict()}


def main(path: Path):
    for p in (RESULTS, FIGS, MODELS):
        p.mkdir(parents=True, exist_ok=True)
    df = load(path)
    print(f"Loaded {len(df):,} Southwest flights")
    ops = ops_analytics(df)
    mdl = model_part(df)
    (RESULTS / "metrics.json").write_text(json.dumps({"ops": ops, "model": mdl}, indent=2, default=float))

    k = ops["kpis"]
    best = next(r for r in mdl["models"] if r["model"] == mdl["best_model"])
    md = [
        "# Results (auto-generated by `src/train.py`)\n",
        "## Operations KPIs\n",
        f"- Flights analyzed: **{k['flights']:,}** across **{k['airports']}** airports, **{k['routes']}** routes",
        f"- On-time arrival rate: **{k['on_time_rate']:.1%}** | Cancellation rate: **{k['cancel_rate']:.2%}**",
        f"- Average arrival delay when late: **{k['avg_delay_min_when_late']:.0f} min**",
        f"- Delay rate before 9 AM: **{ops['morning_vs_evening']['before_9am']:.1%}** vs after 5 PM: "
        f"**{ops['morning_vs_evening']['after_5pm']:.1%}**\n",
        "## Delay minutes by cause (%)\n",
        "\n".join(f"- {c}: **{v}%**" for c, v in ops["cause_share_pct"].items()),
        "\n## Delay prediction (pre-departure features only)\n",
        f"Train: {', '.join(mdl['train_months'])} ({mdl['train_rows']:,} flights) | "
        f"Test: {mdl['test_month']} ({mdl['test_rows']:,} flights, {mdl['test_delay_rate']:.1%} delayed)\n",
        pd.DataFrame(mdl["models"]).to_markdown(index=False, floatfmt=".3f"),
        f"\n**{mdl['best_model']}**: flagging the riskiest 20% of flights captures "
        f"**{best['capture_top20_pct']:.0f}%** of all delayed flights "
        f"({best['capture_top20_pct'] / 20:.1f}x better than random).\n",
        "## Top predictors (permutation importance)\n",
        "\n".join(f"{i+1}. `{f}`" for i, f in enumerate(mdl["top_features"])),
    ]
    (RESULTS / "results.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "southwest_flights.csv")
    main(ap.parse_args().data)
