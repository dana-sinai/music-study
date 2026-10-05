"""Manuscript figures (APA style, 300 dpi). Figure 1: emotion series, published vs corrected.
Figure 2: stream-weighted lyric-theme shares over time (small multiples)."""
from pathlib import Path
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.dates as mdates
R = Path(__file__).parent; F = R / "figures"
BLUE, GRAY, INK, MUTED = "#2a78d6", "#8c8b86", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": .6})
O = pd.read_csv(R.parent / "lyrics_pipeline/data/weekly_emotion_scores.csv", parse_dates=["week_date"]).set_index("week_date")
V = pd.read_csv(R / "results_v2_corpus/weekly_emotion_scores_v2.csv", parse_dates=["week_date"]).set_index("week_date")
T = pd.read_csv(R / "themes/weekly_theme_shares.csv", parse_dates=["week_date"]).set_index("week_date")

def fmt(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7])); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), sharex=True)
for ax, e, lab in zip(axes.flat, ["fear", "disgust", "trust", "sadness"], ["Fear", "Disgust", "Trust", "Sadness"]):
    ax.plot(O.index, O[e], color=GRAY, lw=1.6, ls=(0, (4, 2)), label="Published corpus (408 songs)")
    ax.plot(V.index, V[e], color=BLUE, lw=2, label="Audited corpus (484 songs)")
    ax.set_title(lab, loc="left", fontsize=10, color=INK); ax.set_ylabel("Stream-weighted probability", fontsize=8); fmt(ax)
for ax in axes[1]: plt.setp(ax.get_xticklabels(), rotation=0, fontsize=8)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=8.5, bbox_to_anchor=(.5, -.01))
fig.tight_layout(rect=(0, .05, 1, 1)); fig.savefig(F / "figure2.png", dpi=300, bbox_inches="tight"); plt.close(fig)

themes = [("war_security", "War and security"), ("nation_home", "Nation and homeland"), ("hope_resilience", "Hope and resilience"),
          ("party_hedonism", "Party and drinking"), ("romance_heartbreak", "Romance and heartbreak"), ("faith_prayer", "Faith and prayer")]
fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6), sharex=True, sharey=True)
for ax, (k, lab) in zip(axes.flat, themes):
    y = T[k] * 100
    ax.plot(y.index, y, color=BLUE, lw=.8, alpha=.35)
    ax.plot(y.index, y.rolling(8, center=True, min_periods=4).mean(), color=BLUE, lw=2)
    ax.set_title(lab, loc="left", fontsize=9.5, color=INK); ax.set_ylim(0, 60)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1])); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
for ax in axes[:, 0]: ax.set_ylabel("% of weekly streams", fontsize=8)
for ax in axes[1]: plt.setp(ax.get_xticklabels(), fontsize=7.5)
fig.tight_layout(); fig.savefig(F / "figure1.png", dpi=300, bbox_inches="tight"); plt.close(fig)
print("ok")
