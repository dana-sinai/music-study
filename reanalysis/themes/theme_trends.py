"""Trends in lyric themes, coverage drift, and a validity check of HebEMO 'fear' (v2 corpus)."""
from pathlib import Path
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
TH = Path(__file__).parent
V2 = ROOT / "reanalysis/results_v2_corpus"
W = pd.read_csv(TH / "weekly_theme_shares.csv", parse_dates=["week_date"]).set_index("week_date")
S = pd.read_csv(V2 / "weekly_emotion_scores_v2.csv", parse_dates=["week_date"]).set_index("week_date")
songs = pd.read_csv(TH / "song_themes.csv").merge(pd.read_csv(V2 / "song_emotions_v2.csv"), on="uri")
t = np.arange(len(W)); X = sm.add_constant(t)

def ar1(y):
    r = sm.GLSAR(np.asarray(y, float), X, rho=1).iterative_fit(10)
    return r.params[1] * 52, r.pvalues[1]          # change per year

rows = []
for th in W.columns:
    y = W[th]; b, p = ar1(y)
    early, late = y[:26].mean(), y[-26:].mean()
    rows.append(dict(theme=th, mean_share=y.mean(), first_6mo=early, last_6mo=late,
                     change_per_year=b, p_ar1=p, rho=stats.spearmanr(t, y)[0]))
tr = pd.DataFrame(rows); tr.to_csv(TH / "theme_trends.csv", index=False)
print(tr.round(3).to_string(index=False))

print("\nCoverage drift (share of weekly streams with scored lyrics):")
b, p = ar1(S.stream_coverage); print(f"  first 6 mo {S.stream_coverage[:26].mean():.1%}, last 6 mo {S.stream_coverage[-26:].mean():.1%}, per-year {b:+.3f}, AR1 p={p:.3f}")

print("\nSong-level validity: mean HebEMO fear by theme (Mann-Whitney vs songs without the theme)")
for th in W.columns:
    a, b_ = songs.loc[songs[th] == 1, "fear"], songs.loc[songs[th] == 0, "fear"]
    if len(a) > 4:
        print(f"  {th:20} n={len(a):3}  fear {a.mean():.3f} vs {b_.mean():.3f}  p={stats.mannwhitneyu(a, b_).pvalue:.3f}")
print("\nWeekly: HebEMO fear vs war_security share  r =",
      round(np.corrcoef(S.fear, W.war_security)[0, 1], 2),
      " (differenced r =", round(np.corrcoef(np.diff(S.fear), np.diff(W.war_security))[0, 1], 2), ")")
dec = W[W.index.month == 12].mean(); rest = W[W.index.month != 12].mean()
print("\nDecember vs other months (holiday seasonality):"); print((dec - rest).round(3).to_string())
