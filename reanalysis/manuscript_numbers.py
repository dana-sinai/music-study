"""Every number cited in the v2 manuscript, computed from committed results (no lyrics needed)."""
import json
from pathlib import Path
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats
R = Path(__file__).parent
V2 = R / "results_v2_corpus"
charts = pd.read_csv(R.parent / "lyrics_pipeline/data/spotify_israel_combined.csv", parse_dates=["week_date"])
m = pd.read_csv(V2 / "matches.csv"); se = pd.read_csv(V2 / "song_emotions_v2.csv")
W = pd.read_csv(V2 / "weekly_emotion_scores_v2.csv", parse_dates=["week_date"]).set_index("week_date")
O = pd.read_csv(R.parent / "lyrics_pipeline/data/weekly_emotion_scores.csv", parse_dates=["week_date"]).set_index("week_date")
E = ["joy", "sadness", "anger", "fear", "trust", "disgust", "anticipation"]
t = np.arange(len(W)); X = sm.add_constant(t)
N = {}
tot = charts.streams.sum()
N["weeks"] = int(W.shape[0]); N["entries"] = int(len(charts)); N["tracks"] = int(charts.uri.nunique())
N["israeli_songs"] = int(len(m)); N["status"] = m.status.value_counts().to_dict()
N["status_stream_share"] = (m.groupby("status").total_streams.sum() / tot).round(4).to_dict()
N["scored"] = int(len(se)); N["coverage_mean"] = float(W.stream_coverage.mean())
N["coverage_min"] = float(W.stream_coverage.min()); N["coverage_max"] = float(W.stream_coverage.max())
N["saturation_gt90"] = {e: float((se[e] > .9).mean()) for e in E}
N["saturation_lt10"] = {e: float((se[e] < .1).mean()) for e in E}
N["neg_sent_gt90"] = float((se.sentiment_negative > .9).mean())
N["windowed_songs"] = int((se.n_windows > 1).sum())
def ar1(y):
    r = sm.GLSAR(np.asarray(y, float), X, rho=1).iterative_fit(10); return float(r.params[1]), float(r.pvalues[1]), float(r.model.rho[0])
def hac(y):
    r = sm.OLS(np.asarray(y, float), X).fit(cov_type="HAC", cov_kwds={"maxlags": 12}); return float(r.pvalues[1])
def acf1(y): y = np.asarray(y); return float(np.corrcoef(y[:-1], y[1:])[0, 1])
rows = []
for e in E:
    for lab, S in [("published", O), ("v2", W)]:
        b, p, rho_ar = ar1(S[e])
        rows.append(dict(emotion=e, series=lab, mean=float(S[e].mean()), spearman=float(stats.spearmanr(t, S[e])[0]),
                         p_naive=float(stats.spearmanr(t, S[e])[1]), p_hac=hac(S[e]), p_ar1=p, slope_per_year=b * 52,
                         acf1=acf1(S[e]), first6=float(S[e][:26].mean()), last6=float(S[e][-26:].mean())))
N["trends"] = rows
# placebo & Chow-anywhere (v2 fear and disgust)
rng = np.random.default_rng(1); hits = 0
for _ in range(2000):
    x = np.zeros(124)
    for i in range(1, 124): x[i] = .97 * x[i - 1] + rng.normal()
    r_, p_ = stats.spearmanr(t, x); hits += (abs(r_) > .5 and p_ < .001)
N["placebo_share"] = hits / 2000
def chow(y, k):
    def ssr(z): return sm.OLS(z, sm.add_constant(np.arange(len(z)))).fit().ssr
    s, s1, s2 = ssr(y), ssr(y[:k]), ssr(y[k:]); F = ((s - s1 - s2) / 2) / ((s1 + s2) / (len(y) - 4))
    return stats.f.sf(F, 2, len(y) - 4)
N["chow_anywhere"] = {e: float(np.mean([chow(W[e].values, k) < .001 for k in range(15, 110)])) for e in E}
z = (W.joy.rolling(3, center=True).mean() - W.joy.mean()) / W.joy.std()
N["joy_peak_z"] = float(z.max()); N["joy_peak_week"] = str(z.idxmax().date())
# drivers of joy peak and fear decline (v2)
cm = charts.merge(se[["uri"] + E], on="uri")
win = cm[(cm.week_date >= "2024-10-10") & (cm.week_date < "2025-01-01")]
jc = win.groupby(["track_name", "artist_names"]).apply(lambda x: (x.joy * x.streams).sum() / win.streams.sum()).sort_values(ascending=False)
N["joy_window_mean"] = float((win.joy * win.streams).sum() / win.streams.sum()); N["joy_top_song"] = list(jc.index[0]); N["joy_top_share"] = float(jc.iloc[0] / jc.sum())
def share(d): return d.groupby("uri").streams.sum() / d.streams.sum()
e_, l_ = share(cm[cm.week_date < "2024-04-11"]), share(cm[cm.week_date >= "2025-08-28"]); idx = e_.index.union(l_.index)
d = l_.reindex(idx, fill_value=0) - e_.reindex(idx, fill_value=0); sv = se.set_index("uri")
fc = (d * sv.loc[idx, "fear"]).sort_values(key=abs, ascending=False)
names = charts.drop_duplicates("uri").set_index("uri")
N["fear_change"] = float(fc.sum()); N["fear_top5_share"] = float(fc.head(5).sum() / fc.sum())
N["fear_top5"] = [[names.loc[u, "track_name"], names.loc[u, "artist_names"], float(v)] for u, v in fc.head(5).items()]
# themes
th = pd.read_csv(R / "themes/theme_trends.csv"); N["themes"] = th.to_dict(orient="records")
TW = pd.read_csv(R / "themes/weekly_theme_shares.csv", parse_dates=["week_date"]).set_index("week_date")
N["fear_war_r"] = float(np.corrcoef(W.fear, TW.war_security)[0, 1])
N["fear_war_r_diff"] = float(np.corrcoef(np.diff(W.fear), np.diff(TW.war_security))[0, 1])
st = pd.read_csv(R / "themes/song_themes.csv"); N["theme_song_share"] = st[[c for c in TW.columns]].mean().round(3).to_dict()
cov_b, cov_p, _ = ar1(W.stream_coverage); N["coverage_trend_p"] = cov_p
N["coverage_first6"] = float(W.stream_coverage[:26].mean()); N["coverage_last6"] = float(W.stream_coverage[-26:].mean())
json.dump(N, open(R / "manuscript_numbers.json", "w"), indent=1, ensure_ascii=False, default=str)
print(json.dumps({k: N[k] for k in ["status", "status_stream_share", "scored", "coverage_mean", "neg_sent_gt90", "windowed_songs",
      "placebo_share", "chow_anywhere", "joy_peak_z", "joy_top_song", "joy_top_share", "fear_change", "fear_top5_share", "fear_top5",
      "fear_war_r", "fear_war_r_diff", "coverage_trend_p"]}, ensure_ascii=False, indent=0, default=str))
print(pd.DataFrame(N["trends"]).round(3).to_string(index=False))
print({k: round(v, 2) for k, v in N["saturation_gt90"].items()})
