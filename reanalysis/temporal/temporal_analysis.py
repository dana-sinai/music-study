"""
Temporal shape and anchor-event analysis of the weekly emotion and theme series (audited corpus).

1. Trajectories: describe the shape of each series (half-year means, smoothed peak/trough) and test whether a
   non-linear (spline) trajectory fits better than a straight line, with AR(1) errors.
2. Anchor events: for each documented event, change from the 3 weeks before to the 3 weeks from the event on.
   Its p value comes from the same change computed at every other possible week (a permutation-in-time null),
   which keeps the autocorrelation of the series. For notable shifts, the songs that produced them.

Usage: python reanalysis/temporal/temporal_analysis.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrix
from statsmodels.nonparametric.smoothers_lowess import lowess

HERE = Path(__file__).parent
R = HERE.parent
CHARTS = R.parent / "lyrics_pipeline/data/spotify_israel_combined.csv"

EMO = ["joy", "sadness", "anger", "fear", "trust", "disgust", "anticipation"]
THEMES = ["war_security", "grief_loss", "faith_prayer", "hope_resilience", "nation_home", "romance_heartbreak",
          "party_hedonism"]
EVENTS = [  # from the earlier manuscript, plus the October 2025 peace deal
    ("a", "2023-10-07", "October 7 attack"),
    ("b", "2024-04-13", "Iranian attack (True Promise I)"),
    ("c", "2024-06-08", "Nuseirat hostage rescue"),
    ("d", "2024-09-17", "Pager attacks on Hezbollah"),
    ("e", "2024-10-01", "Iranian missile barrage (True Promise II)"),
    ("f", "2024-10-17", "Sinwar killed"),
    ("g", "2024-11-27", "Israel–Hezbollah ceasefire"),
    ("h", "2025-01-19", "Gaza ceasefire takes effect"),
    ("i", "2025-03-18", "Fighting in Gaza resumes"),
    ("j", "2025-06-13", "12-day war with Iran (Rising Lion)"),
    ("k", "2025-10-09", "Gaza peace deal signed; hostages released Oct 13"),
]
K = 3  # weeks before / after


def load():
    E = pd.read_csv(R / "results_v2_corpus/weekly_emotion_scores_v2.csv", parse_dates=["week_date"]).set_index("week_date")
    T = pd.read_csv(R / "themes/weekly_theme_shares.csv", parse_dates=["week_date"]).set_index("week_date")
    return pd.concat([E[EMO], T[THEMES]], axis=1)


def half_years(W):
    edges = pd.to_datetime(["2023-10-01", "2024-04-01", "2024-10-01", "2025-04-01", "2025-10-01", "2026-03-01"])
    labels = ["Oct 2023–Mar 2024", "Apr–Sep 2024", "Oct 2024–Mar 2025", "Apr–Sep 2025", "Oct 2025–Feb 2026"]
    g = pd.cut(W.index, edges, labels=labels, right=False)
    return W.groupby(g, observed=True).mean()


def shape_test(y):
    """GLS with AR(1) errors: linear vs natural cubic spline (4 df). Wald test of the extra spline terms."""
    t = np.arange(len(y)) / (len(y) - 1)
    Xs = np.asarray(dmatrix("cr(t, df=4) - 1", {"t": t}, return_type="dataframe"))
    Xl = sm.add_constant(t)
    fit_s = sm.GLSAR(y.values, Xs, rho=1).iterative_fit(maxiter=20)
    # nested test: spline space contains the line; compare via the AR(1)-whitened residual sums of squares
    rho = fit_s.model.rho
    def wssr(X):
        yw = y.values[1:] - rho * y.values[:-1]
        Xw = X[1:] - rho * X[:-1]
        b, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        return ((yw - Xw @ b) ** 2).sum(), Xw.shape[1]
    s_l, k_l = wssr(Xl)
    s_s, k_s = wssr(Xs)
    n = len(y) - 1
    F = ((s_l - s_s) / (k_s - k_l)) / (s_s / (n - k_s))
    from scipy.stats import f as fdist
    return float(F), float(fdist.sf(F, k_s - k_l, n - k_s))


def smooth(y, frac=.2):
    x = np.arange(len(y))
    return pd.Series(lowess(y.values, x, frac=frac, return_sorted=False), index=y.index)


def event_index(W, date):
    d = pd.Timestamp(date)
    pos = np.searchsorted(W.index.values, d.to_datetime64())  # first chart week ending on/after the event
    return int(pos)


def window_change(y, i):
    if i - K < 0 or i + K > len(y):
        return np.nan
    return y.iloc[i:i + K].mean() - y.iloc[i - K:i].mean()


def event_table(W):
    rows = []
    for col in W.columns:
        y = W[col]
        null = np.array([window_change(y, i) for i in range(K, len(y) - K + 1)])
        sd = y.std()
        for code, date, label in EVENTS:
            i = event_index(W, date)
            ch = window_change(y, i)
            if np.isnan(ch):
                rows.append(dict(series=col, event=code, label=label, week=str(W.index[min(i, len(W) - 1)].date()),
                                 change=np.nan, change_sd=np.nan, p_perm=np.nan, level_z=float((y.iloc[i:i + K].mean() - y.mean()) / sd)))
                continue
            p = float((np.abs(null) >= abs(ch)).mean())
            rows.append(dict(series=col, event=code, label=label, week=str(W.index[i].date()), change=float(ch),
                             change_sd=float(ch / sd), p_perm=p, level_z=float((y.iloc[i:i + K].mean() - y.mean()) / sd)))
    return pd.DataFrame(rows)


def song_drivers(series, i, charts, scores, top=3):
    """Songs whose change in stream share x score produced the before->after change."""
    weeks = sorted(charts.week_date.unique())
    before, after = weeks[i - K:i], weeks[i:i + K]
    c = charts.merge(scores[["uri", series]], on="uri")
    def contrib(ws):
        s = c[c.week_date.isin(ws)]
        tot = s.groupby("week_date").streams.sum()
        s = s.assign(w=s.streams / s.week_date.map(tot))
        return (s.w * s[series]).groupby([s.uri, s.track_name, s.artist_names]).sum() / len(ws)
    d = contrib(after).sub(contrib(before), fill_value=0)
    tot = d.sum()
    d = d.reindex(d.abs().sort_values(ascending=False).index)[:top]
    return [dict(track=t, artist=a, contribution=float(v), share=float(v / tot) if tot else np.nan) for (u, t, a), v in d.items()]


def main():
    W = load()
    out = {}
    hy = half_years(W)
    hy.to_csv(HERE / "half_year_means.csv")
    shapes = []
    for col in W.columns:
        y = W[col]
        F, p = shape_test(y)
        s = smooth(y)
        shapes.append(dict(series=col, F_nonlinear=F, p_nonlinear=p, peak_week=str(s.idxmax().date()),
                           peak=float(s.max()), trough_week=str(s.idxmin().date()), trough=float(s.min()),
                           start=float(s.iloc[0]), end=float(s.iloc[-1])))
    shapes = pd.DataFrame(shapes)
    shapes.to_csv(HERE / "trajectory_shapes.csv", index=False)

    ev = event_table(W)
    ev.to_csv(HERE / "event_windows.csv", index=False)

    charts = pd.read_csv(CHARTS, parse_dates=["week_date"])
    se = pd.read_csv(R / "results_v2_corpus/song_emotions_v2.csv")
    st = pd.read_csv(R / "themes/song_themes.csv")
    scores = se.merge(st, on="uri", how="outer")
    notable = ev[(ev.p_perm < .05)].sort_values("p_perm")
    drivers = []
    for r in notable.itertuples():
        i = event_index(W, dict((c, d) for c, d, _ in EVENTS)[r.event])
        drivers.append(dict(series=r.series, event=r.event, label=r.label, change=r.change, p=r.p_perm,
                            songs=song_drivers(r.series, i, charts, scores)))
    (HERE / "event_drivers.json").write_text(json.dumps(drivers, ensure_ascii=False, indent=1), encoding="utf-8")

    pd.set_option("display.width", 200)
    print("Half-year means:\n", (hy * 1).round(3).T.to_string())
    print("\nTrajectory shape (spline vs line, AR(1) errors):\n", shapes.round(4).to_string(index=False))
    print("\nEvent windows with permutation p < .10:\n",
          ev[ev.p_perm < .10].sort_values(["event", "p_perm"]).round(4).to_string(index=False))
    print("\nDrivers of p < .05 shifts:")
    for d in drivers:
        print(f"  [{d['event']}] {d['label']} – {d['series']} change {d['change']:+.4f} (p={d['p']:.3f})")
        for s in d["songs"]:
            print(f"       {s['track']} – {s['artist']}: {s['contribution']:+.4f} ({s['share']:.0%})")


if __name__ == "__main__":
    main()
