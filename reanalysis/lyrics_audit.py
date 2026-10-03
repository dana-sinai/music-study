"""
Audit of the Shironet lyric matching (05_shironet_scraper_with_mapping.py).

Inputs (reanalysis/data/, not committed; from Drive music_analysis/output/):
  spotify_israel_combined.csv, hebrew_songs_with_lyrics.csv,
  artist_hebrew_mapping_combined.csv, song_emotions.csv

Writes reanalysis/lyrics_review_queue.csv: every unmatched or suspect song,
ranked by streams, with empty columns for manual correction.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

HERE = Path(__file__).parent
DATA = HERE / "data"
E = ["joy", "sadness", "anger", "fear", "trust", "disgust", "anticipation"]
# Flagged by the artist check but verified as correct: transliterations and covers
VERIFIED_OK = {"הכל יהיה", "תהיה בנאדם", "ירושלים", "לילות וקללות", "לתת ולקחת", "עטלף עיוור"}

charts = pd.read_csv(DATA / "spotify_israel_combined.csv", parse_dates=["week_date"])
lyr = pd.read_csv(DATA / "hebrew_songs_with_lyrics.csv")
amap = pd.read_csv(DATA / "artist_hebrew_mapping_combined.csv")
songs = pd.read_csv(DATA / "song_emotions.csv").drop_duplicates("spotify_uri")
mp = dict(zip(amap.original_artist, amap.hebrew_artist))
total = charts.streams.sum()


def name_variants(artists):
    out = {str(artists)}
    for key in [artists] + [p.strip() for p in str(artists).split(",")]:
        out.add(str(key))
        if pd.notna(mp.get(key)):
            out |= {x.strip() for x in str(mp[key]).split(",")}
    return out


def artist_sim(spotify_artists, shironet_artist):
    return max(max(fuzz.partial_ratio(v, str(shironet_artist)), fuzz.token_set_ratio(v, str(shironet_artist)))
               for v in name_variants(spotify_artists))


print("== Coverage ==")
heb = charts[charts.is_hebrew]
print(f"unique chart tracks {charts.uri.nunique()}; Hebrew-titled {heb.uri.nunique()} "
      f"({heb.streams.sum()/total:.1%} of streams); with lyrics {lyr.lyrics.notna().sum()}")
missing = lyr[lyr.lyrics.isna()]
print(f"Hebrew-titled songs without lyrics: {len(missing)} ({missing.total_streams.sum()/total:.1%} of all streams)")

matched = lyr[lyr.lyrics.notna()].copy()
matched["artist_sim"] = [artist_sim(a, s) for a, s in zip(matched.artist_names, matched.shironet_artist)]
suspect = matched[(matched.artist_sim < 70) & ~matched.track_name.isin(VERIFIED_OK)]
print(f"\n== Wrong-song matches (Shironet artist does not match) ==\n{len(suspect)} of {len(matched)} matched songs")
dup = matched[matched.duplicated("shironet_url", keep=False)]
print(f"Shironet pages reused for >1 Spotify track: {dup.shironet_url.nunique()} (mostly versions/covers)")
long_ = (matched.lyrics.str.len() > 1500).sum()
print(f"lyrics likely truncated at 512 tokens (>1500 chars): {long_}")

print("\n== Impact of dropping wrong-song matches on weekly series ==")
m = charts.merge(songs[["spotify_uri"] + E], left_on="uri", right_on="spotify_uri")
def weekly(df):
    return df.groupby("week_date").apply(
        lambda x: pd.Series({e: (x[e] * x.streams).sum() / x.streams.sum() for e in E}))
A, B = weekly(m), weekly(m[~m.uri.isin(suspect.spotify_uri)])
t = np.arange(len(A))
from scipy import stats
for e in E:
    print(f"  {e:13} r(orig,clean)={np.corrcoef(A[e], B[e])[0,1]:.2f}  "
          f"trend rho {stats.spearmanr(t, A[e])[0]:+.2f} -> {stats.spearmanr(t, B[e])[0]:+.2f}")

queue = pd.concat([
    missing.assign(issue="no lyrics found"),
    suspect.assign(issue="wrong artist on Shironet"),
])[["issue", "spotify_uri", "track_name", "artist_names", "total_streams", "peak_rank",
    "shironet_title", "shironet_artist", "shironet_url"]]
queue = queue.sort_values("total_streams", ascending=False)
queue["cum_share_of_problem_streams"] = (queue.total_streams.cumsum() / queue.total_streams.sum()).round(3)
for col in ["correct_shironet_url", "status", "notes"]:
    queue[col] = ""
queue.to_csv(HERE / "lyrics_review_queue.csv", index=False, encoding="utf-8-sig")
print(f"\nWrote lyrics_review_queue.csv ({len(queue)} rows). Top 50 rows = "
      f"{queue.cum_share_of_problem_streams.iloc[49]:.0%} of problem streams.")
