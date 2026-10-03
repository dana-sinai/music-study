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
