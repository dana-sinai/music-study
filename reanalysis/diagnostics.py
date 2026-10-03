"""
Robustness diagnostics for the stream-weighted lyric-emotion series
(reanalysis after the SPPE rejection, Oct 2026).

Inputs (place in reanalysis/data/, not committed):
  spotify_israel_combined.csv  - weekly Spotify Top-200 Israel, Oct 2023 - Feb 2026
  song_emotions.csv            - HebEMO probabilities per song (408 songs)
Both live in Drive: music_analysis/output/.

Checks:
  1. Classifier saturation at the song level
  2. Rebuild the weekly series (should match weekly_emotion_scores.csv exactly)
  3. Autocorrelation and autocorrelation-robust trend tests
  4. Placebo: how often pure AR(0.97) noise gives |rho|>0.5, p<.001
  5. Chow test at every candidate week (are the event dates special?)
  6. Data-driven change points vs the paper's event-anchored phases
  7. Song-level drivers: joy surge (Oct 2024) and early->late trend decomposition
"""
from pathlib import Path
import numpy as np
import pandas as pd
import ruptures as rpt
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.stattools import acf

DATA = Path(__file__).parent / "data"
E = ["joy", "sadness", "anger", "fear", "trust", "disgust", "anticipation"]

charts = pd.read_csv(DATA / "spotify_israel_combined.csv", parse_dates=["week_date"])
songs = pd.read_csv(DATA / "song_emotions.csv").drop_duplicates("spotify_uri")
m = charts.merge(songs[["spotify_uri"] + E], left_on="uri", right_on="spotify_uri")


def weekly(df):
    return df.groupby("week_date").apply(
        lambda x: pd.Series({e: (x[e] * x.streams).sum() / x.streams.sum() for e in E}))


def chow_p(y, k):
    def ssr(yy):
        return sm.OLS(yy, sm.add_constant(np.arange(len(yy)))).fit().ssr
    s, s1, s2 = ssr(y), ssr(y[:k]), ssr(y[k:])
    f = ((s - (s1 + s2)) / 2) / ((s1 + s2) / (len(y) - 4))
    return stats.f.sf(f, 2, len(y) - 4)


print("1) Song-level saturation (share of songs)")
for e in E:
    v = songs[e]
    print(f"  {e:13} >0.9 {(v > .9).mean():4.0%}   <0.1 {(v < .1).mean():4.0%}")
print(f"  sentiment_negative > 0.9: {(songs.sentiment_negative > .9).mean():.0%} of songs")

W = weekly(m)
t = np.arange(len(W))
X = sm.add_constant(t)
print(f"\n2) Rebuilt series: {len(W)} weeks, {m.uri.nunique()} matched songs, {len(m)} chart entries")

print("\n3) Trend tests")
print(f"  {'emotion':13}{'acf1':>6}{'Neff':>6}{'rho':>7}{'naive p':>10}{'HAC p':>8}{'AR(1) p':>9}")
for e in E:
    y = W[e].values
    a = acf(y, nlags=1)[1]
    rho = stats.spearmanr(t, y)[0]
    p_ols = sm.OLS(y, X).fit().pvalues[1]
    p_hac = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 12}).pvalues[1]
    p_ar = sm.GLSAR(y, X, rho=1).iterative_fit(10).pvalues[1]
    print(f"  {e:13}{a:6.2f}{len(y)*(1-a)/(1+a):6.1f}{rho:7.2f}{p_ols:10.1e}{p_hac:8.3f}{p_ar:9.3f}")

rng = np.random.default_rng(1)
hits = 0
for _ in range(2000):
    x = np.zeros(len(t))
    for i in range(1, len(t)):
        x[i] = 0.97 * x[i - 1] + rng.normal()
    r, p = stats.spearmanr(t, x)
    hits += abs(r) > .5 and p < .001
print(f"\n4) Placebo: AR(0.97) noise reaches |rho|>0.5, p<.001 in {hits/2000:.0%} of runs")

print("\n5) Chow test at every week 15..109: share 'significant' at p<.001")
for e in E:
    y = W[e].values
    print(f"  {e:13} {np.mean([chow_p(y, k) < .001 for k in range(15, 110)]):.0%}")

print("\n6) Data-driven change points (Binseg, piecewise-linear, 3 breaks)")
for e in E:
    y = ((W[e] - W[e].mean()) / W[e].std()).values
    bk = rpt.Binseg(model="linear").fit(np.column_stack([y, np.ones_like(t), t])).predict(n_bkps=3)[:-1]
    print(f"  {e:13}", [str(W.index[b].date()) for b in bk])
print("  paper:        2024-04-13, 2024-09-17, 2025-06-13")

print("\n7a) Joy, Oct-Dec 2024: contribution by song")
win = m[(m.week_date >= "2024-10-10") & (m.week_date < "2025-01-01")]
contrib = win.groupby(["track_name", "artist_names"]).apply(
    lambda x: (x.joy * x.streams).sum() / win.streams.sum()).sort_values(ascending=False)
print(contrib.head(5).round(4).to_string())
top = contrib.index[0][0]
for df, lab in [(W, "all songs"), (weekly(m[m.track_name != top]), f"without {top}")]:
    z = (df.joy.rolling(3, center=True).mean() - df.joy.mean()) / df.joy.std()
    print(f"  max joy z ({lab}): {z.max():.2f} at {z.idxmax().date()}")

print("\n7b) Early (<Apr 2024) -> late (>=Oct 2025) change, concentration across songs")
def share(df):
    return df.groupby("uri").streams.sum() / df.streams.sum()
se, sl = share(m[m.week_date < "2024-04-01"]), share(m[m.week_date >= "2025-10-01"])
idx = se.index.union(sl.index)
d = sl.reindex(idx, fill_value=0) - se.reindex(idx, fill_value=0)
sv = songs.set_index("spotify_uri")
for e in E:
    c = (d * sv.loc[idx, e]).sort_values(key=abs, ascending=False)
    print(f"  {e:13} change {c.sum():+.3f}; top-5 songs account for {c.head(5).sum()/c.sum():.0%}")
