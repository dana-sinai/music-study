# Lyrics pipeline v2 (Shironet matching + HebEMO scoring)

Rebuilds the lyric corpus from scratch, fixing what the Feb 2026 run got wrong
(see `../REVISION_PLAN.md`, "Lyric matching audit").

| Problem in the Feb 2026 run | Fix here |
|---|---|
| Accept rule `0.8·title + 0.2·artist > 70` let a different artist's song through on a title match (43 wrong songs) | Artist is a **hard gate**; a title match with the wrong artist goes to review, never to auto-accept |
| Search hits were read with `get_text(strip=True)`, so "יהיה טוב" became "יהיהטוב" | Words keep their spaces, and comparison also uses a space-free form |
| 164 searches logged as "0 results" (bot-protection pages) | Challenge pages are detected and retried with backoff; every page is cached on disk |
| Live, acoustic and medley titles were never found | Version tags are stripped; medleys are split into their songs and the lyrics joined |
| Songs selected only if the *title* is in Hebrew | Also selects Israeli artists with English-letter titles; lyrics that are not Hebrew are excluded at scoring and listed |
| Artists missing from the Hebrew-name table could never match | Their Shironet artist ID is learned from ≥2 different exact-title hits; once confirmed, that artist ID is also used |
| 49 long songs truncated at 512 tokens | Overlapping 512-token windows, averaged |
| One `song_emotions.csv` reused across 211- and 408-song versions | A single run writes a single, frozen `out/` corpus |

Replaying the new decision rule on the 408 old matches: **0 of the 43 wrong songs auto-accepted**
(38 sent to review, 5 rejected). Of the 365 correct ones, 351 are auto-accepted and 14 go to review
(mostly covers, e.g. Eyal Golan singing Hanan Ben Ari's song, which a human should confirm).

## Run

Needs network access to `shironet.mako.co.il` (matching) and `huggingface.co` (scoring, once).

```bash
pip install -r requirements.txt
mkdir -p data   # copy in: spotify_israel_combined.csv, artist_hebrew_mapping_combined.csv,
                #          weekly_emotion_scores.csv (old, for comparison)
python -m pytest -q tests             # parser + matching-rule tests (offline)
python run_match.py --limit 20        # smoke test
python run_match.py                   # ~840 songs, ~1.5-2 h at 2.5 s/request
# hand-check out/review_queue.csv, put decisions in data/manual_overrides.csv, then:
python run_match.py --offline         # re-decide from cache + overrides, no new requests
python score_hebemo.py                # -> out/song_emotions_v2.csv
python build_series.py                # -> out/weekly_emotion_scores_v2.csv + comparison
```

`data/manual_overrides.csv` (format in `manual_overrides.example.csv`): `action` is
`url` (use this Shironet page) or `none` (no lyrics exist). Overrides always win.

**Do not commit `out/`**: it holds copyrighted lyrics and cached pages. Share song IDs,
Shironet URLs and scores only.

## Validation still to do by people
1. Work through `out/review_queue.csv` (sorted by streams).
2. A second person checks a random 10% of auto-accepted matches; report the agreement.
