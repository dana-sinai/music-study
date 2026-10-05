"""Figures for the temporal/event analysis.
Figure 2: weekly z-scores of each emotion (thin) with a LOWESS trajectory (thick) and anchor events a–k.
Figure 3: monthly heatmap of emotions and themes (row z-scores), like the earlier manuscript's Figure 1."""
from pathlib import Path
import sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.dates as mdates
from matplotlib.colors import TwoSlopeNorm
sys.path.insert(0, str(Path(__file__).parent))
from temporal_analysis import load, smooth, EVENTS, EMO, THEMES

F = Path(__file__).parents[1] / "figures"
BLUE, INK, MUTED, EV = "#2a78d6", "#0b0b0b", "#52514e", "#b23a3a"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
W = load()
Z = (W - W.mean()) / W.std()
LAB = {"joy": "Joy", "sadness": "Sadness", "anger": "Anger", "fear": "Fear", "trust": "Trust", "disgust": "Disgust",
       "anticipation": "Anticipation", "war_security": "War and security", "grief_loss": "Grief and loss",
       "faith_prayer": "Faith and prayer", "hope_resilience": "Hope and resilience", "nation_home": "Nation and homeland",
       "romance_heartbreak": "Romance and heartbreak", "party_hedonism": "Partying and drinking"}

def events(ax, labels=False):
    for code, d, _ in EVENTS:
        x = pd.Timestamp(d)
        if x < W.index[0] - pd.Timedelta(days=7):
            continue
        ax.axvline(x, color=EV, lw=.7, ls=(0, (2, 2)), alpha=.8, zorder=0)
        if labels:
            ax.text(x, 1.02, code, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=7.5, color=EV)

def panel_fig(cols, name, ncol=2):
    n = len(cols); nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.2, 1.55 * nrow + .4), sharex=True, sharey=True)
    axes = axes.flat
    for k, (ax, c) in enumerate(zip(axes, cols)):
        z = Z[c]
        ax.axhline(0, color="#bdbcb8", lw=.6)
        ax.plot(z.index, z, color=BLUE, lw=.7, alpha=.35)
        ax.plot(z.index, smooth(z), color=BLUE, lw=2)
        events(ax, labels=k < ncol)
        ax.set_title(LAB[c], loc="left", fontsize=9, color=INK, pad=10 if k < ncol else 3)
        ax.set_ylim(-3.2, 3.2)
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7])); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
        ax.tick_params(labelsize=7.5)
    for ax in list(axes)[n:]:
        ax.set_visible(False)
    fig.supylabel("z score", fontsize=8.5)
    fig.tight_layout(); fig.savefig(F / name, dpi=300, bbox_inches="tight"); plt.close(fig)

panel_fig(EMO, "supp_figure_emotions_lowess.png")

# Figure 2: the earlier manuscript's z-score figure, redrawn from the audited series:
# one coloured panel per emotion, weekly z score as a filled area, anchor events labelled on the top panel.
COLORS = {"joy": "#FFD700", "sadness": "#4169E1", "anger": "#DC143C", "fear": "#8B008B", "trust": "#32CD32",
          "disgust": "#8B4513", "anticipation": "#FF8C00"}
SHORT = {"a": "Oct 7 attack", "b": "Iran attack (True Promise I)", "c": "Nuseirat rescue", "d": "Pager attacks",
         "e": "Iran barrage (True Promise II)", "f": "Sinwar killed", "g": "Hezbollah ceasefire",
         "h": "Gaza ceasefire", "i": "Gaza war resumes", "j": "12-day war (Rising Lion)", "k": "Gaza peace deal"}
fig, axes = plt.subplots(len(EMO), 1, figsize=(11, 2.0 * len(EMO) + 1.6), sharex=True, sharey=True)
for k, (ax, e) in enumerate(zip(axes, EMO)):
    z = Z[e]
    ax.fill_between(z.index, z, 0, color=COLORS[e], alpha=.55, linewidth=0)
    ax.plot(z.index, z, color=COLORS[e], lw=1.6)
    ax.axhline(0, color=INK, lw=.6, alpha=.4)
    for code, d, _ in EVENTS:
        x = max(pd.Timestamp(d), W.index[0])
        ax.axvline(x, color="#6b6a66", lw=.8, ls=(0, (3, 2)), alpha=.7, zorder=0)
        if k == 0:
            ax.text(x, 1.04, f"({code}) {SHORT[code]}", transform=ax.get_xaxis_transform(), rotation=60,
                    ha="left", va="bottom", fontsize=7.5, color=INK)
    ax.set_ylim(-3.2, 3.2)
    ax.set_ylabel("z", fontsize=8.5)
    ax.text(.005, .9, LAB[e], transform=ax.transAxes, fontsize=10, fontweight="bold", color=INK, va="top")
    ax.text(.995, .9, f"M = {W[e].mean():.3f}, SD = {W[e].std():.3f}".replace("0.", "."), transform=ax.transAxes,
            ha="right", va="top", fontsize=7.5, color=MUTED)
    ax.grid(axis="y", color="#e6e5e0", lw=.6)
axes[-1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
axes[-1].tick_params(labelsize=8)
fig.tight_layout(h_pad=.4)
fig.savefig(F / "figure2.png", dpi=300, bbox_inches="tight"); plt.close(fig)
panel_fig(THEMES, "supp_figure_themes_z.png")

# heatmap
M = Z.resample("MS").mean()
rows = EMO + THEMES
fig, ax = plt.subplots(figsize=(7.2, 4.4))
im = ax.imshow(M[rows].T.values, aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(0, -2, 2), interpolation="nearest")
ax.set_yticks(range(len(rows))); ax.set_yticklabels([LAB[r] for r in rows], fontsize=8)
ax.axhline(len(EMO) - .5, color="white", lw=3)
xt = [i for i, d in enumerate(M.index) if d.month in (1, 4, 7, 10)]
ax.set_xticks(xt); ax.set_xticklabels([M.index[i].strftime("%b %y") for i in xt], fontsize=7.5)
for code, d, _ in EVENTS:
    t = pd.Timestamp(d)
    if t < M.index[0]:
        continue
    pos = (t.year - M.index[0].year) * 12 + t.month - M.index[0].month + (t.day - 1) / 30 - .5
    ax.axvline(pos, color=INK, lw=.6, ls=(0, (2, 2)))
    ax.text(pos, -.7, code, ha="center", va="bottom", fontsize=7.5, color=INK)
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=.025, pad=.01); cb.set_label("Monthly mean z score", fontsize=8); cb.ax.tick_params(labelsize=7)
fig.tight_layout(); fig.savefig(F / "figure3.png", dpi=300, bbox_inches="tight"); plt.close(fig)
