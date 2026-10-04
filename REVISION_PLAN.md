# Revision plan: "A population affective barometer" (rejected, SPPE, 3 Sep 2026)

The decision came with no reviewer comments. This plan comes from re-running the analysis
(`reanalysis/diagnostics.py`, which reproduces the published weekly series exactly) and
reading the manuscript as a critical reviewer would.

## 1. What the reanalysis shows

| Claim in the manuscript | What the data show |
|---|---|
| HebEMO measures the emotion of each song | Outputs are close to binary. 98% of songs score >0.9 on *negative sentiment*; 76% score >0.9 on disgust; only 7 of 408 songs score >0.5 on joy. Most Israeli pop is love songs, so this is a domain-shift failure (the model was trained on COVID-era Facebook comments), not a finding. |
| Strong monotonic trends (e.g., disgust ρ = .84, P < .001) | The weekly series are close to random walks (lag-1 autocorrelation 0.89–0.97, effective N ≈ 2–7). Pure AR(0.97) noise reaches \|ρ\| > .5 at P < .001 in 34% of simulations. After an AR(1) correction only **disgust** (p = .007), **fear** (p = .004) and, weakly, **trust** (p = .027) keep a trend. Sadness (p = .32) and anticipation (p = .72) do not. |
| Chow tests "validated" the event-anchored phase boundaries | The test is uninformative here: a break placed at *any* week is "significant" at p < .001 in 85–100% of cases for six of the seven emotions. Data-driven change points do not line up with the chosen dates. |
| A "manic cluster": a joy surge after the pager attacks / Sinwar (z = +2.69) | The surge is **one song**: "תמיד אוהב אותי" (ששון איפרם שאולוב), which entered the chart on 2024-09-26 and reached #1. It supplies about 63% of weighted joy in Oct–Dec 2024. Without it the October peak disappears, and the series maximum moves to May 2024 (z = 2.0). The plateau-then-slow-decay shape is a hit song's chart lifecycle. |
| Sadness rose across the war | 73% of the early-to-late rise comes from 5 songs. The fear decline comes entirely from 5 songs. Disgust is the only broad-based shift (top 5 songs = 13%). |
| 211 unique songs analysed | The scored file and the chart merge both contain **408** matched songs (14,522 entries). One of the two numbers in the manuscript is wrong. |

Framing problems a reviewer will flag:
- "Clinically predicted before the data were examined" contradicts the Introduction (the idea came from noticing the music) and the analysis history (10-period and 4-period schemes, plus a dropped PTSD correlation). This needs to go.
- The title and conclusion claim a proxy for *mental health*, but no mental-health outcome was measured.
- "Manic defence", TMT and "emotional numbing" are interpretations of a signal that has not been shown to be valid.
- There is no pre-war baseline, so seasonality, holidays, Memorial Day, and ordinary chart churn can't be separated from war effects.
- Supply and demand are confounded: new war-themed releases (supply) are read as listener need (demand).
- Coverage: about 33% of streams are unscored, including all non-Hebrew songs (22% of streams), and the missing streams are not random.
- Citation errors: Many Labs 4 is attributed to ref 14 (Pyszczynski 2006); ref 2 is cited for a WHO priority statement.

### What the Colab notebook adds (`music_analysis.ipynb`; the copy `music_analysis 2.ipynb` is identical)

- **The song count is explained, and the paper mixes two data versions.** The HebEMO run saved in the notebook scored **211** songs. The paper's Table 1 descriptives come from that run. The lyrics file later grew to 408 songs, and every time-series result (Tables 2–3, z-scores) comes from the **408**-song file: Table 3 reproduces exactly from it. Re-run once on a single, frozen corpus and report one N.
- **The structural-break (Chow) and Kruskal-Wallis tests are not in the notebook or the Drive scripts.** No four-phase code exists either; the notebook uses a 2-period split, and `analysis_summary.py` uses 10 periods. Find the code that produced these results (local Mac folder?) or re-derive them before resubmitting. The reported joy z = +2.69 also doesn't match the notebook (max z = 2.99).
- **The original hypotheses contradict the paper's "predicted" claim.** The notebook's pre-stated H1 was that "sadness and anger peak in Q4 2023–Q1 2024, then decline". The data show sadness *rising*. The "manic defence index" came from the psychoanalytic essay that started the project, and the joy song (תמיד אוהב אותי) was listed as an essay song in advance.
- **Songs are cut off at 512 tokens.** Long lyrics are truncated, and repeated choruses are scored as they appear. Report how many songs were truncated.
- **An external-criterion test already exists and was dropped.** Monthly PTSD incidence from the MHRC (Krivoy-Charite data) covers Oct 2023–Oct 2024 (13 months). The October 2024 value is entered as **30** in one cell and **19** in another and in `analysis_summary.py`; check the source. Of 8 emotions × 3 lags plus 2 indices, **trust** stands out (r = −0.82). It **survives linear detrending** (r = −0.82 with 30; −0.72 with 19) and partly survives differencing (r = −0.69 / −0.54). Other emotions do not. **Update after the lyric audit (below): this association disappears once the wrong-song matches are removed** (raw r = −0.35; detrended r = +0.32). It was produced by wrong lyrics, not by listening. Re-test only after the corpus is fixed.

### Lyric matching audit (`reanalysis/lyrics_audit.py`)

How the pipeline worked: songs were selected if the Spotify *title* contained Hebrew characters (818 of 1,318 tracks, 78% of streams). Shironet was then searched by title only, and the best hit was accepted when 0.8 × title similarity + 0.2 × artist similarity exceeded 70.

1. **A wrong artist can still pass.** With a perfect title match, the score is ≥ 80 even when the artist similarity is 0. **43 of 408 matched songs (11%; 7% of scored streams) carry another artist's song with the same title.** Examples: "אהבה" by Osher Cohen (Rain Sobotka's lyrics), "השם ירחם" by Tuna (Itay Zvulun's), "רוזה" by Omer Adam (Yehoram Gaon's), and "צוחקת ובוכה" by Eden Hason (Ilanit's). These include the main **trust** songs (הריני, צוחקת ובוכה), a top **fear** song (עוד יום), and one of the 7 **joy** songs (פרפר).
   - Effect of removing them: weekly trust correlates only r = .73 with the original series. The disgust trend drops from ρ = .84 to .63, and the anger trend disappears.
2. **Half of the Hebrew songs have no lyrics.** 410 of 818 Hebrew-titled songs (8.4% of all streams) are missing, including #1 hits ("אחת ממיליון", "לאהוב אותך כל יום", "רוקי"). 34 of them are live, acoustic or medley versions whose decorated titles fail the search ("- Live", "גרסה אקוסטית", "&", "(7.10.23)").
3. **Selection by title script misses some Israeli songs.** 15 songs with English-letter titles are excluded (e.g., "Hurricane", "New Day Will Rise"). This is only 0.8% of streams, but it is not random: these are Eurovision songs.
4. **Long lyrics are truncated.** 49 songs (>1,500 characters) were cut at 512 tokens.
5. **Correct lyrics now cover only 79% of streams from Hebrew-titled songs.**

**Fixes:**
- Make the artist a **hard gate**: accept a hit only if the Shironet performer matches the mapped Hebrew artist, or if it is a documented cover.
- Search by title + artist, and fall back to the artist's Shironet song list.
- Normalize titles before searching: strip version tags; split medleys into their component songs.
- Select songs by artist or language, not by title script.
- Score long lyrics in chunks and average the scores, instead of truncating.
- Hand-check the ranked `reanalysis/lyrics_review_queue.csv`. The top 50 rows cover 66% of problem streams; the top 100 cover 82%.
- Then have a second person verify a random 10% of all matches and report the agreement rate.
- Freeze the corpus and re-run *everything* on it once.

**Implemented in `lyrics_pipeline/`** (tested offline; needs network access to Shironet and Hugging Face to run). Replaying its decision rule on the old matches auto-accepts none of the 43 wrong songs. The old log also showed that **164 searches returned "0 results"** because of bot-protection pages, not missing songs, and that a parser bug glued title words together.

## 2. Redesign

**Main question (revised):** Does a stream-weighted, *validated* index of the emotional content in what Israelis
listen to track independent measures of population distress, before and during the war?

1. **Measurement validation (required).**
   - Have 3 native-speaker raters score a stratified sample (~120 songs) on valence, arousal and 4–6 discrete emotions. Report inter-rater reliability (ICC).
   - Compare candidate scorers against the raters: HebEMO, a multilingual or Hebrew model fine-tuned on lyrics, and an LLM rater with a fixed prompt. Use the scorer that agrees best with the humans, with its agreement reported.
   - Add audio features (e.g., Essentia or musicnn valence/arousal from previews), because listeners choose music largely for sound, not lyrics.
   - Use continuous dimensions (valence/arousal) instead of 8 binary classifiers.
2. **Design.**
   - Add a **pre-war baseline**: weekly Israel charts from 2019 onward, so Oct 7 can be analysed as an **interrupted time series** with seasonality and holiday controls.
   - Add **controls**: the non-Hebrew part of the same chart, and a comparison country's chart.
3. **Supply versus demand.** Split each week's signal into
   - (a) streams of catalog songs that already existed before Oct 7, which is a cleaner demand signal, and
   - (b) new releases.
4. **External criterion (makes or breaks a psychiatry journal).** Correlate the index with independent weekly or monthly series:
   - Clalit indicators via Ofer's team: psychiatric ED visits, new anxiolytic or antidepressant prescriptions.
   - ERAN or NATAL hotline volumes.
   - Survey waves: Levin et al.'s cohort; IDI/INSS mood and optimism items.
   - Google Trends for distress terms.

   Use pre-whitened or differenced cross-correlations, not raw correlations of trending series.
5. **Statistics.**
   - ARIMA or ITS models with HAC errors.
   - Data-driven change points with confidence intervals.
   - Leave-one-song-out and top-k-songs-out influence analysis on every headline result.
   - FDR correction across emotions.
   - Sensitivity to coverage.
6. **Prospective test.** Pre-register (OSF) predictions for data *after* Feb 2026 before looking at it. This turns the descriptive story into a real test.
7. **Open science.** Release code and derived scores. Do **not** redistribute Shironet lyrics; share song IDs and scores only. Keep `.spotify_credentials.json` out of any public repo.

## 3. Venue

| Route | Paper | Venues |
|---|---|---|
| **A: methods/descriptive, fast (≈1–2 months)** | Validation (step 1) + baseline ITS (step 2) + an honest robustness section. Reframed as "what lyric-emotion analytics can and cannot reveal about wartime listening". | Royal Society Open Science (published the weather–music paper; values transparency); EPJ Data Science; Music & Science; Psychology of Music. |
| **B: barometer claim, stronger (≈4–6 months)** | Route A + external criterion (step 4), ideally with Clalit data. | JMIR Mental Health / JMIR Infodemiology; Social Science & Medicine – Mental Health; back to SPPE-tier psychiatric epidemiology only with a clear link to distress. |

Recommendation: start with Route A and plan for B. Do not send the current version through the Springer transfer offer.

## 4. Preprint

Yes, but **after** the core fixes (steps 1, 2 and 5), not the current version. Once posted, a preprint is permanent and cited, and the current "manic cluster" claim would not survive a public reanalysis.
- Server: **PsyArXiv** (fits the music-psychology and computational-social-science readership). Use **medRxiv** instead if Route B with clinical data is the lead.
- All venues listed above accept preprinted manuscripts. Check each journal's current policy before posting.
- Post together with the OSF pre-registration of the prospective test (step 6) to establish priority.

### Re-scored corrected corpus (4 Oct 2026) — `reanalysis/results_corrected_old_corpus/`

Shironet blocked the cloud server (HTTP 403) after ~120 pages and stayed blocked for 14+ h, so the new
matching run could not finish. Instead, the **old lyrics minus the 51 wrong/doubtful matches** (357 songs kept;
1 non-Hebrew skipped) were re-scored with windowed HebEMO (12 long songs no longer truncated).
Weekly stream coverage falls from 66.7% to **60.9%** (range 37–73%).

| Emotion | trend p, published series (AR(1)-corrected) | trend p, corrected corpus |
|---|---|---|
| fear (falling) | .004 | **< .001** |
| disgust (rising) | .007 | .146 |
| trust (falling) | .027 | .236 |
| sadness (rising) | .322 | .147 |
| joy, anger, anticipation | n.s. | n.s. |

- **Only the decline in fear survives.** The disgust and trust trends depended on the wrong lyrics.
- The Oct 2024 joy peak is unchanged (z = 2.95). It was always one correctly matched song (תמיד אוהב אותי), so it is still a single-song effect.
- The classifier still saturates (98% of songs scored "negative"). Measurement validity (step 1 of the redesign) remains the main problem, whatever happens with matching.
