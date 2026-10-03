"""Rebuild the weekly stream-weighted series from the new corpus and compare with the
published (2026-02) version.

Usage: python build_series.py   -> out/weekly_emotion_scores_v2.csv, printed comparison
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).parent
DATA, OUT = HERE / "data", HERE / "out"
E = ["joy", "sadness", "anger", "fear", "trust", "disgust", "anticipation"]


def weekly(charts, scores):
    m = charts.merge(scores, left_on="uri", right_on="uri")
    cov = m.groupby("week_date").streams.sum() / charts.groupby("week_date").streams.sum()
    w = m.groupby("week_date").apply(lambda x: pd.Series({e: np.average(x[e], weights=x.streams) for e in E}))
    w["stream_coverage"] = cov
    return w


def main():
    charts = pd.read_csv(DATA / "spotify_israel_combined.csv", parse_dates=["week_date"])
    new = pd.read_csv(OUT / "song_emotions_v2.csv")
    W = weekly(charts, new)
    W.to_csv(OUT / "weekly_emotion_scores_v2.csv")
    print(f"songs scored: {len(new)}; mean weekly stream coverage {W.stream_coverage.mean():.1%} "
          f"(published version: 66.7%)")
    old_p = DATA / "weekly_emotion_scores.csv"
    if old_p.exists():
        old = pd.read_csv(old_p, parse_dates=["week_date"]).set_index("week_date")
        t = np.arange(len(W))
        print(f"\n{'emotion':13}{'r(old,new)':>11}{'rho old':>9}{'rho new':>9}")
        for e in E:
            print(f"{e:13}{np.corrcoef(old[e], W[e])[0,1]:11.2f}"
                  f"{stats.spearmanr(t, old[e])[0]:9.2f}{stats.spearmanr(t, W[e])[0]:9.2f}")


if __name__ == "__main__":
    main()
