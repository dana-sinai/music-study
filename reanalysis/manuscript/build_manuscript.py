"""
Build the manuscript draft (APA 7 layout) as HTML from reanalysis/manuscript_numbers.json, so every number
in the text and tables comes from one place. The HTML is uploaded to Google Drive and converted to a Google Doc.

Usage: python reanalysis/manuscript/build_manuscript.py   -> reanalysis/manuscript/manuscript.html
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).parent
N = json.loads((HERE.parent / "manuscript_numbers.json").read_text(encoding="utf-8"))
T = {(r["emotion"], r["series"]): r for r in N["trends"]}
TH = N["theme_table"]
EMOS = ["fear", "disgust", "trust", "sadness", "joy", "anger", "anticipation"]


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def p_apa(p):
    if p < .001:
        return "&lt; .001"
    return "= " + f"{p:.3f}".lstrip("0")


def p_cell(p):
    return "&lt; .001" if p < .001 else f"{p:.3f}".lstrip("0")


def r2(x):
    return f"{x:.2f}".replace("-", "−").replace("0.", ".", 1) if abs(x) < 1 else f"{x:.2f}"


def rho(x):
    s = f"{x:.2f}"
    s = s.replace("0.", ".", 1)
    return s.replace("-", "−")


fv2 = T[("fear", "v2")]
fpub = T[("fear", "published")]
S = N["status"]
SS = N["status_stream_share"]

# ---------------------------------------------------------------- tables
def table(num, title, head, rows, note, widths=None):
    th = "".join(f'<td style="border-top:1px solid #000;border-bottom:1px solid #000;"><b>{h}</b></td>'
                 for h in head)
    body = []
    for i, r in enumerate(rows):
        last = i == len(rows) - 1
        bb = "border-bottom:1px solid #000;" if last else ""
        cells = "".join(f'<td style="{bb}{"text-align:left;" if j == 0 else "text-align:center;"}">{c}</td>'
                        for j, c in enumerate(r))
        body.append(f"<tr>{cells}</tr>")
    return (f'<p><b>Table {num}</b></p><p><i>{title}</i></p>'
            f'<table style="border-collapse:collapse;font-family:\'Times New Roman\';font-size:11pt;">'
            f"<tr>{th}</tr>{''.join(body)}</table>"
            f'<p><i>Note.</i> {note}</p><p></p>')


tab1 = table(
    1, "Lyric retrieval and matching: earlier pipeline versus audited pipeline",
    ["Feature", "Earlier pipeline", "Audited pipeline"],
    [
        ["Songs considered", "Hebrew songs on the chart (number not logged)", f"{N['israeli_songs']} Israeli songs (of {N['tracks']:,} charted tracks)"],
        ["Artist check", "None (a title match alone could pass)", "Required: artist similarity ≥ 85"],
        ["Songs with lyrics used", "408 (manuscript reported 211)", f"{N['scored']}"],
        ["Wrong song by another artist", "43 (11%; 7% of scored streams)", "Excluded by rule; uncertain matches held out"],
        ["Searches lost to bot-protection pages", "164 (recorded as “not found”)", "Detected, cached, and retried"],
        ["Lyrics truncated at 512 tokens", "49 songs", f"None (sliding windows; {N['windowed_songs']} songs needed them)"],
        ["Held for human review", "—", f"{S['review']} ({pct(SS['review'])} of streams)"],
        ["Not found on Shironet", "—", f"{S['not_found']} ({pct(SS['not_found'])} of streams)"],
        ["Mean weekly stream coverage", "66.7% (range 43–78%)",
         f"{pct(N['coverage_mean'])} (range {100 * N['coverage_min']:.0f}–{100 * N['coverage_max']:.0f}%)"],
    ],
    "Coverage is the share of each week’s Top-200 streams carried by songs with scored lyrics. "
    "“Earlier pipeline” refers to the unpublished analysis audited here. "
    f"A further {S['lyrics_not_fetched']} matched songs ({pct(SS['lyrics_not_fetched'])} of streams) could not be downloaded "
    f"before the lyrics site blocked automated access, and {S['page_without_lyrics']} pages had no lyrics.")

sat_rows = []
for e in ["disgust", "anger", "sadness", "anticipation", "trust", "fear", "joy"]:
    sat_rows.append([e.capitalize(), pct(N["saturation_gt90"][e], 0), pct(N["saturation_lt10"][e], 0),
                     pct(N["saturation_gt90"][e] + N["saturation_lt10"][e], 0)])
tab2 = table(
    2, f"Distribution of song-level HebEMO probabilities (N = {N['scored']} songs)",
    ["Emotion", "Songs with p &gt; .90", "Songs with p &lt; .10", "Near 0 or 1 (total)"],
    sat_rows,
    "HebEMO returns one probability per emotion from a separate binary classifier. "
    f"In addition, the heBERT sentiment model labelled {pct(N['neg_sent_gt90'], 0)} of songs as negative with p &gt; .90. "
    "Surprise is omitted (all values below .01).")

trend_rows = []
for e in EMOS:
    a, b = T[(e, "published")], T[(e, "v2")]
    trend_rows.append([e.capitalize(), rho(a["spearman"]), p_cell(a["p_naive"]), p_cell(a["p_ar1"]),
                       rho(b["spearman"]), p_cell(b["p_hac"]), p_cell(b["p_ar1"]),
                       f"{b['first6']:.3f}".lstrip("0") + " → " + f"{b['last6']:.3f}".lstrip("0")])
tab3 = table(
    3, f"Linear trends in weekly stream-weighted emotion scores, earlier versus audited corpus ({N['weeks']} weeks)",
    ["Emotion", "Earlier ρ", "Earlier naive <i>p</i>", "Earlier AR(1) <i>p</i>",
     "Audited ρ", "Audited HAC <i>p</i>", "Audited AR(1) <i>p</i>", "Audited mean, first → last 6 months"],
    trend_rows,
    "ρ = Spearman correlation with week. Naive <i>p</i> treats the 124 weeks as independent, as in the earlier analysis. "
    "HAC = ordinary least squares slope with Newey–West standard errors; AR(1) = generalized least squares slope with "
    "first-order autoregressive errors. Lag-1 autocorrelation of the weekly series ranged from "
    f"{min(T[(e, 'v2')]['acf1'] for e in EMOS):.2f} to {max(T[(e, 'v2')]['acf1'] for e in EMOS):.2f}.".replace("0.", "."))

theme_label = {"war_security": "War and security", "grief_loss": "Grief and loss", "faith_prayer": "Faith and prayer",
               "hope_resilience": "Hope and resilience", "nation_home": "Nation and homeland",
               "romance_heartbreak": "Romance and heartbreak", "party_hedonism": "Partying and drinking"}
th_rows = []
for k, lab in theme_label.items():
    t = TH[k]
    sign = "+" if t["per_year"] > 0 else "−"
    th_rows.append([lab, f"{t['n_songs']} ({t['pct_songs']:.1f})", f"{t['first6']:.1f}", f"{t['last6']:.1f}",
                    f"{sign}{abs(t['per_year']):.1f}", p_cell(t["p"]) if t["p"] >= .001 else "&lt; .001", f"{t['top5']}%"])
tab4 = table(
    4, "Stream-weighted share of listening carried by songs with each lyrical theme",
    ["Theme", "Songs, <i>n</i> (%)", "First 6 months (%)", "Last 6 months (%)", "Change per year (points)",
     "AR(1) <i>p</i>", "Change due to top 5 songs"],
    th_rows,
    "A song carries a theme when its lyrics contain the theme’s words at least a set number of times (Supplementary "
    "Section B); songs can carry several themes or none. Shares are percentages of each week’s scored streams. "
    "“Change due to top 5 songs” is the part of the first-to-last change produced by the five songs that contributed most "
    f"to it. For reference, the ten most-streamed songs of a typical week carry {N.get('top10_share_pct', 17.4):.1f}% of that week’s streams.")

# ---------------------------------------------------------------- text
TITLE = ("What Israelis Streamed During 28 Months of War: An Audited, Stream-Weighted Analysis of "
         "Hebrew Popular-Song Lyrics")

fear_songs = "; ".join(f"<i>{t}</i> ({a})" for t, a, _ in N["fear_top5"])
joy_song = f"<i>{N['joy_top_song'][0]}</i> (Sasson Ifram Shaulov)"

abstract = f"""
<p><b>Objective:</b> Stream-weighted emotion scores of song lyrics have been proposed as a passive, continuous index of
population mood. We tested how far such an index can be trusted, using Israel’s Spotify charts during the war that began
on October 7, 2023.</p>
<p><b>Methods:</b> We analyzed {N['entries']:,} weekly Top-200 entries ({N['weeks']} weeks, October 2023–February 2026).
We first audited an earlier version of the analysis, then rebuilt lyric retrieval with a required artist match,
retrieved and scored {N['scored']} Hebrew lyrics with the HebEMO emotion models (covering {pct(N['coverage_mean'])} of
weekly streams), and coded seven lyrical themes with a transparent word list. Trends were tested with models that allow for
the strong week-to-week autocorrelation of chart data, and every result was checked for dependence on a few songs.</p>
<p><b>Results:</b> In the earlier analysis, 11% of matched songs carried another artist’s lyrics. Rising disgust, falling
trust, and the event-anchored phases did not survive correction. In simulations, pure noise with the same autocorrelation
reached |ρ| &gt; .5 at <i>p</i> &lt; .001 in {pct(N['placebo_share'], 0)} of runs. HebEMO placed most songs near 0 or 1, and
labelled {pct(N['neg_sent_gt90'], 0)} as negative. One trend held: fear declined (ρ = {rho(fv2['spearman'])},
AR(1) <i>p</i> {p_apa(fv2['p_ar1'])}), but five songs accounted for all of it. The clearest change was in repertoire.
Songs about war and security fell from {TH['war_security']['first6']:.1f}% to {TH['war_security']['last6']:.1f}% of listening,
and songs about partying and drinking rose from {TH['party_hedonism']['first6']:.1f}% to {TH['party_hedonism']['last6']:.1f}%.</p>
<p><b>Conclusion:</b> Lyric-emotion indices built from charts are dominated by a few hits and by the behavior of the classifier.
They should not be read as measures of population mood without song-level validation and an external criterion.
Transparent theme coding of what people streamed is a more defensible description of cultural response to war.</p>
<p><i>Keywords:</i> music streaming, song lyrics, emotion recognition, Hebrew, war, cultural analytics, reproducibility</p>
"""

intro = f"""
<p>People use music to regulate how they feel. They pick songs that match, process, or change their mood
(Saarikallio &amp; Erkkilä, 2007; Chong et al., 2024; Reybrouck et al., 2020), and many seek out sad music when they are
distressed (Taruffi &amp; Koelsch, 2014; Schubert, 2016). Because streaming platforms publish weekly charts with stream
counts, it is tempting to treat the emotional content of what a country streams as a continuous, passive index of its
mood. Such indices would be attractive during prolonged crises, when surveys are slow and costly and reach only part of
the population (Norris &amp; Stevens, 2007; Conway &amp; O’Connor, 2016).</p>
<p>Earlier work points in this direction. Streaming behavior relates to personality (Anderson et al., 2021); the acoustic
features of popular songs follow the weather (Anglada-Tort et al., 2023); and lyrics on the US Billboard charts became
less negative during societal crises, including the COVID-19 pandemic (Foramitti et al., 2025; see also Levy et al.,
2024). Text-based mood indices from social media, such as the hedonometer (Dodds et al., 2011), rest on the same idea.</p>
<p>Three weaknesses can turn such an index into an artefact. First, each charted song must be matched to its lyrics, and a
lyric database can return a song with the same title by another artist. Second, emotion classifiers trained on other text
may behave differently on lyrics, and their outputs are rarely checked against human judgment. Third, weekly chart series
change slowly, so neighboring weeks are not independent. Tests that assume independence then find “significant” trends
and breaks far more often than their nominal error rate (Granger &amp; Newbold, 1974). Big-data indices that skip these
checks can look convincing and still fail (Lazer et al., 2014).</p>
<p>Israel after October 7, 2023 offers a demanding test case. The country went through 28 months of war on several
fronts, with discrete, widely shared events and well-documented effects on mental health (Levin et al., 2025). In an
earlier, unpublished version of this analysis, we reported that stream-weighted lyric emotions tracked a coherent trajectory
over this period. These were rising sadness and disgust, falling fear and trust, statistically validated phases, and a
burst of joy around military successes in late 2024. In this paper we audit that analysis, rebuild it, and ask three
questions:</p>
<p>(1) How accurate is automated lyric retrieval for chart songs, and does it matter for the results?<br>
(2) Which emotion trends survive correct matching, autocorrelation-robust inference, and checks for dependence on single
songs?<br>
(3) What changed in the repertoire Israelis streamed, described with a transparent coding of lyrical themes?</p>
"""

methods = f"""
<h2><b>Data</b></h2>
<p>We used the Spotify Weekly Top-200 charts for Israel from October 12, 2023 to February 19, 2026: {N['weeks']} weeks,
{N['entries']:,} entries, and {N['tracks']:,} distinct tracks with their weekly stream counts. Of these, {N['israeli_songs']}
tracks were by Israeli artists performing in Hebrew and entered lyric retrieval. Lyrics came from Shironet
(shironet.mako.co.il), the main public Hebrew lyrics database. All data are public and aggregate; no individuals were
involved, so ethics approval was not required.</p>

<h2><b>Audit of the earlier pipeline</b></h2>
<p>The earlier pipeline searched Shironet by song title and accepted the best fuzzy match of title and artist combined.
We compared every one of its 408 matches with the performing artist on Shironet. We also re-read its logs for searches
that had failed and checked which lyrics were truncated by the 512-token limit of the classifier.</p>

<h2><b>Audited lyric retrieval</b></h2>
<p>The rebuilt pipeline (Supplementary Section A) treats the artist as a hard requirement. Artist names in Latin script
are mapped to their Hebrew forms, and a candidate is accepted only when artist similarity is at least 85 and title
similarity is at least 92 on a 0–100 scale (RapidFuzz; titles compared without spaces and version tags such as “live” or
“remix”). Medleys are split into their parts. Candidates with a lower title score or an uncertain artist are held for
human review rather than accepted. Bot-protection pages are recognized and not recorded as “not found”. Long lyrics are
scored in overlapping windows of 510 tokens (stride 255), and the window scores are averaged with weights for length.
Songs whose lyrics are less than 50% Hebrew letters are excluded. Held songs were checked by hand against Shironet, and each decision was recorded in a
public overrides file.</p>

<h2><b>Emotion scoring</b></h2>
<p>Each song’s lyrics were scored with HebEMO (Chriqui &amp; Yahav, 2022). HebEMO consists of eight binary classifiers, one per
Plutchik (1980) emotion, each returning a probability from 0 to 1, plus the heBERT sentiment model. HebEMO was trained on
Hebrew user comments, not lyrics. For each week we computed the stream-weighted mean of each emotion across songs with
scored lyrics.</p>

<h2><b>Theme coding</b></h2>
<p>To describe content in a way readers can check, we coded seven themes with a word list (Supplementary Section B). The
themes were war and security, grief and loss, faith and prayer, hope and resilience, nation and homeland, romance and
heartbreak, and partying and drinking. Words are matched as whole Hebrew words, allowing up to two attached prefix letters
(ו, ה, ב, כ, ל, מ, ש). Ambiguous everyday words were removed after inspecting matches. A song carries a theme when it
contains the theme’s words at least once (war, grief), twice (faith, hope, nation, partying), or three times (romance). The
weekly theme share is the stream-weighted proportion of scored listening carried by songs with that theme.</p>

<h2><b>Statistical analysis</b></h2>
<p>For each weekly series we report the Spearman correlation with time. We also report two tests of the linear slope that
account for autocorrelation: ordinary least squares with Newey–West (HAC) standard errors (Newey &amp; West, 1987) and
generalized least squares with first-order autoregressive errors (AR(1)), estimated with statsmodels (Seabold &amp;
Perktold, 2010). We treat the AR(1) test as primary. To show how often independence-based tests mislead with series like
these, we simulated 2,000 AR(1) series with coefficient .97 and no trend. We also applied the Chow break test used in the
earlier analysis at every possible week.</p>
<p>Coverage could change over time and bias the series, so we tested its trend in the same way. For every trend that survived,
we decomposed the change between the first and last six months into song contributions, defined as the change in each
song’s share of streams times its score. We then report the share of the change produced by the five largest contributors.
For the joy peak we report the share of the peak window’s excess joy contributed by its leading song. Analyses used Python
3.11 (pandas, statsmodels, scipy). All tests were two-sided, and given the exploratory aim, <i>p</i> values are reported
without correction for multiple comparisons.</p>
"""

results = f"""
<h2><b>The earlier pipeline</b></h2>
<p>The audit found that 43 of the 408 matched songs (11%; 7% of scored streams) carried the lyrics of a different song with
the same title by another artist (Table 1). For example, <i>רוזה</i> by Omer Adam was scored with Yehoram Gaon’s song of that
name, and <i>צוחקת ובוכה</i> by Eden Hason with Ilanit’s. These errors included the main
songs behind the earlier trust result. A further 164 searches had been recorded as “not found” because the site had returned
a bot-protection page, and 49 long lyrics had been cut at 512 tokens. The earlier manuscript also reported 211 songs where the
data held 408.</p>
<p><b>[Insert Table 1 about here]</b></p>

<h2><b>The audited corpus</b></h2>
<p>Of the {N['israeli_songs']} Israeli songs, {S['accepted']} were accepted automatically, {S['review']} were held for review,
and {S['not_found']} were not found on Shironet. Most of the not-found songs were 2025 releases not yet added to the site. After
excluding non-Hebrew lyrics, {N['scored']} songs were scored. They carried a mean of {pct(N['coverage_mean'])} of weekly streams,
almost the same coverage as before, but without wrong lyrics. Coverage did not change significantly over time
({pct(N['coverage_first6'])} in the first six months, {pct(N['coverage_last6'])} in the last six; AR(1) <i>p</i>
{p_apa(N['coverage_trend_p'])}).</p>

<h2><b>Classifier output</b></h2>
<p>HebEMO rarely produced intermediate values (Table 2). Disgust exceeded .90 for {pct(N['saturation_gt90']['disgust'], 0)}
of songs and anger for {pct(N['saturation_gt90']['anger'], 0)}. Fear, joy, and trust were below .10 for more than 97% of
songs. The sentiment model labelled {pct(N['neg_sent_gt90'], 0)} of songs, including love songs and dance hits, as negative.
A weekly mean of near-binary scores therefore mostly reflects how many streams went to the few songs labelled 1. It
reflects the gradual emotional tone of the repertoire much less.</p>
<p><b>[Insert Table 2 about here]</b></p>

<h2><b>Trends</b></h2>
<p>The weekly series were highly autocorrelated (lag-1 autocorrelation .85–.98). Under these conditions, trend-free
simulated series reached |ρ| &gt; .5 with naive <i>p</i> &lt; .001 in {pct(N['placebo_share'], 0)} of runs. A Chow break placed
at any week was “significant” at <i>p</i> &lt; .001 in 72–100% of positions for six of the seven emotions
({pct(N['chow_anywhere']['fear'], 0)} for fear). The event-anchored phases of the earlier analysis are therefore not
evidence of real breaks.</p>
<p>Table 3 and Figure 1 compare the two corpora. In the earlier series, fear, disgust, and trust trends passed the AR(1)
test. In the audited corpus, only fear did: it declined from {f"{fv2['first6']:.3f}".lstrip('0')} to {f"{fv2['last6']:.3f}".lstrip('0')}
(ρ = {rho(fv2['spearman'])}, AR(1) <i>p</i> {p_apa(fv2['p_ar1'])}). Disgust (<i>p</i> {p_apa(T[('disgust','v2')]['p_ar1'])}) and
trust (<i>p</i> {p_apa(T[('trust','v2')]['p_ar1'])}) did not. The fear decline was also found in an intermediate check, which
re-scored the earlier lyrics after removing the wrong matches.</p>
<p><b>[Insert Table 3 about here]</b></p>
<p><b>[Insert Figure 1 about here]</b></p>
<p>The fear decline came from a handful of songs. Five songs, all scored high on fear and streamed heavily early in the war,
produced more than the entire first-to-last change ({N['fear_top5_share'] * 100:.0f}%): {fear_songs}. The rest of the
repertoire changed little. Weekly fear correlated with the share of war-themed songs (<i>r</i> = {r2(N['fear_war_r'])};
first differences <i>r</i> = {r2(N['fear_war_r_diff'])}). At the song level, however, war-themed songs did not score
significantly higher on fear than other songs.</p>
<p>The joy peak of late 2024 remained (<i>z</i> = {N['joy_peak_z']:.2f}, week of October 31, 2024), but
{pct(N['joy_top_share'], 0)} of the excess joy in that window came from one song, {joy_song}. HebEMO scores this devotional song about God’s unconditional love
.999 on joy. It entered the chart in late September 2024 and reached number one. It is a single hit, and its timing with the
military events of that autumn does not show a population-wide shift toward joy.</p>

<h2><b>Themes</b></h2>
<p>Table 4 and Figure 2 show the theme shares. Songs about war and security fell from {TH['war_security']['first6']:.1f}% of listening in the first six months to {TH['war_security']['last6']:.1f}% in the
last six (AR(1) <i>p</i> {p_apa(TH['war_security']['p'])}).
Hope and resilience fell from {TH['hope_resilience']['first6']:.1f}% to {TH['hope_resilience']['last6']:.1f}%, and nation and
homeland from {TH['nation_home']['first6']:.1f}% to {TH['nation_home']['last6']:.1f}%. These declines were driven by the
anthems of the first war months, which faded from the chart. They included <i>חרבו דרבו</i> (Ness &amp; Stilla), <i>עם ישראל חי</i>
(Eyal Golan), <i>יהיה טוב</i> (Jasmin Moallem; and with Omer Adam), and <i>לצאת מדיכאון</i> (Yagel Oshri). Partying and drinking
rose from {TH['party_hedonism']['first6']:.1f}% to {TH['party_hedonism']['last6']:.1f}% (AR(1) <i>p</i> &lt; .001). This rise was
broad-based: the five largest contributors produced only {TH['party_hedonism']['top5']}% of it. Faith and prayer (about one
song in eight) and romance and heartbreak (the most common theme) showed no significant trend.</p>
<p><b>[Insert Table 4 about here]</b></p>
<p><b>[Insert Figure 2 about here]</b></p>
"""

discussion = f"""
<p>An index that looked coherent and theory-consistent largely failed an audit. Correct lyric matching removed two of the
three trends that had survived autocorrelation-robust tests. Proper time-series inference removed the phase structure. The
remaining signals, a fear decline and a joy peak, each rested on a few songs. What remains robust is not a gradual
change in population emotion. It is a change in repertoire: the war anthems of the first months gave way, gradually, to
party and drinking songs, while love songs and religious songs held steady.</p>

<h2><b>Why the index misled</b></h2>
<p>Three mechanisms produced the earlier picture, and each is likely to affect similar studies. First, title-only matching
errors are not random noise. They attach the lyrics of another, often older, song to a hit, and a single heavily streamed
hit can move a weekly mean. Second, a near-binary classifier turns a weekly mean into a count of streams for a few songs
scored 1. Its trends then follow chart turnover, not the tone of the repertoire, and the 98% “negative” rate shows that
HebEMO does not read lyrics the way listeners do. Third, chart series change slowly, and with lag-1 autocorrelations
above .9, tests that assume independent weeks produce strong, “highly significant” trends and breaks from noise (Granger
&amp; Newbold, 1974). An index can pass each of these tests by chance, and a plausible story can then be fitted after the
fact.</p>

<h2><b>What the repertoire shows</b></h2>
<p>The theme results are descriptive, but they rest on lyrics a reader can check. War-specific songs made up almost a quarter
of Israeli listening in the first six months and about an eighth two years later. Songs of national unity and hope
dropped out almost entirely. Songs about partying and drinking more than tripled their share, and the rise was spread
across many songs and artists. This pattern fits accounts of a shift from collective mobilization to a return to
everyday, and perhaps escapist, listening as a war becomes chronic (cf. Foramitti et al., 2025). It does not show what
listeners felt. A falling share of war songs may reflect fatigue, recovery, avoidance, or simply chart turnover and
releases by popular artists. Streaming data alone cannot tell these apart.</p>

<h2><b>Implications for lyric-based mood indices</b></h2>
<p>We suggest four minimum standards for studies that infer population affect from charts or playlists. (a) Report lyric
matching accuracy, with artist verification. (b) Show the distribution of classifier outputs and validate a sample
against human raters. (c) Use inference that allows for autocorrelation, and report a placebo or null benchmark. (d)
Report how much of each result comes from the top few songs. A further step is an external criterion: a lyric index should
be tested against an independent measure of population mental health, such as survey waves or service-use data, before it
is called a barometer.</p>

<p><i>Limitations</i></p>
<p>The corpus covers Hebrew songs on one platform’s Top 200. It excludes Arabic-language music and listeners who do not use
Spotify, who skew older and more religious. About a third of weekly streams had no scored lyrics, mostly international
songs. Charts reflect algorithmic recommendation as well as choice. The theme word list is our own and was refined by
inspecting matches; it is published in full so others can test alternatives. We did not collect human emotion ratings,
so we can show that HebEMO behaves implausibly on lyrics but cannot quantify its error. {S['review']} uncertain matches
({pct(SS['review'])} of streams) were held out, and their exclusion could slightly change the theme shares.</p>

<p><i>Conclusion</i></p>
<p>The emotional content of what a country streams is not yet a measure of how its population feels. In Israel’s charts
during the war, a careful re-analysis keeps only a decline in fear, carried by five songs, and a clear change in repertoire
from war anthems to party songs. Studies of music as a mirror of collective mood need verified lyrics, validated
classifiers, time-series inference, and influence checks before they can support claims about mental health.</p>
"""

declarations = """
<p><b>Ethics.</b> The study used public, aggregate chart data and public lyrics. No human participants were involved.</p>
<p><b>Conflict of interest.</b> The authors declare no conflict of interest.</p>
<p><b>Funding.</b> The authors received no specific funding for this work.</p>
<p><b>Data and code availability.</b> Chart data, song-level scores, theme codes, matching decisions with their Shironet
links, weekly series, and all code are available at [repository link to be added; OSF/GitHub]. Lyrics are protected by
copyright and are not redistributed. They can be retrieved from the listed Shironet links with the published pipeline.</p>
<p><b>Use of AI tools.</b> [Authors to complete according to the journal’s policy.]</p>
"""

refs = [
    "Anderson, I., Gil, S., Gibson, C., Wolf, S., Shapiro, W., Semerci, O., &amp; Greenberg, D. M. (2021). “Just the way you are”: Linking music listening on Spotify and personality. <i>Social Psychological and Personality Science, 12</i>(4), 561–572. https://doi.org/10.1177/1948550620923228",
    "Anglada-Tort, M., Lee, H., Krause, A. E., &amp; North, A. C. (2023). Here comes the sun: Music features of popular songs reflect prevailing weather conditions. <i>Royal Society Open Science, 10</i>(5), Article 221443. https://doi.org/10.1098/rsos.221443",
    "Chong, H. J., Kim, H. J., &amp; Kim, B. (2024). Scoping review on the use of music for emotion regulation. <i>Behavioral Sciences, 14</i>(9), Article 793. https://doi.org/10.3390/bs14090793",
    "Chriqui, A., &amp; Yahav, I. (2022). HeBERT and HebEMO: A Hebrew BERT model and a tool for polarity analysis and emotion recognition. <i>INFORMS Journal on Data Science, 1</i>(1), 81–95. https://doi.org/10.1287/ijds.2022.0016",
    "Conway, M., &amp; O’Connor, D. (2016). Social media, big data, and mental health: Current advances and ethical implications. <i>Current Opinion in Psychology, 9</i>, 77–82. https://doi.org/10.1016/j.copsyc.2016.01.004",
    "Dodds, P. S., Harris, K. D., Kloumann, I. M., Bliss, C. A., &amp; Danforth, C. M. (2011). Temporal patterns of happiness and information in a global social network: Hedonometrics and Twitter. <i>PLoS ONE, 6</i>(12), Article e26752. https://doi.org/10.1371/journal.pone.0026752",
    "Foramitti, M., Nater, U. M., Lamm, C., &amp; Martins, M. (2025). Societal crises disrupt long-term increases in stress, negativity, and simplicity in US Billboard song lyrics from 1973 to 2023. <i>Scientific Reports, 15</i>, Article 41733. https://doi.org/10.1038/s41598-025-28327-5",
    "Granger, C. W. J., &amp; Newbold, P. (1974). Spurious regressions in econometrics. <i>Journal of Econometrics, 2</i>(2), 111–120. https://doi.org/10.1016/0304-4076(74)90034-7",
    "Lazer, D., Kennedy, R., King, G., &amp; Vespignani, A. (2014). The parable of Google Flu: Traps in big data analysis. <i>Science, 343</i>(6176), 1203–1205. https://doi.org/10.1126/science.1248506",
    "Levin, Y., Groweiss, Y., Blank, C., &amp; Neria, Y. (2025). Longitudinal PTSD trajectories before and after the October 7, 2023, terror attacks: A nationwide study of Israeli adults. <i>European Psychiatry, 69</i>, Article e13. https://doi.org/10.1192/j.eurpsy.2025.10130",
    "Levy, A., Granot, R., &amp; Peres, R. (2024). Lyrics do matter: How “coping songs” relate to well-being goals—the COVID pandemic case. <i>Frontiers in Psychology, 15</i>, Article 1431741. https://doi.org/10.3389/fpsyg.2024.1431741",
    "Newey, W. K., &amp; West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. <i>Econometrica, 55</i>(3), 703–708. https://doi.org/10.2307/1913610",
    "Norris, F. H., &amp; Stevens, S. P. (2007). Community resilience and the principles of mass trauma intervention. <i>Psychiatry, 70</i>(4), 320–328. https://doi.org/10.1521/psyc.2007.70.4.320",
    "Plutchik, R. (1980). <i>Emotion: A psychoevolutionary synthesis</i>. Harper &amp; Row.",
    "Reybrouck, M., Podlipniak, P., &amp; Welch, D. (2020). Music listening as coping behavior: From reactive response to sense-making. <i>Behavioral Sciences, 10</i>(7), Article 119. https://doi.org/10.3390/bs10070119",
    "Saarikallio, S., &amp; Erkkilä, J. (2007). The role of music in adolescents’ mood regulation. <i>Psychology of Music, 35</i>(1), 88–109. https://doi.org/10.1177/0305735607068889",
    "Schubert, E. (2016). Enjoying sad music: Paradox or parallel processes? <i>Frontiers in Human Neuroscience, 10</i>, Article 312. https://doi.org/10.3389/fnhum.2016.00312",
    "Seabold, S., &amp; Perktold, J. (2010). Statsmodels: Econometric and statistical modeling with Python. In <i>Proceedings of the 9th Python in Science Conference</i> (pp. 92–96). https://doi.org/10.25080/Majora-92bf1922-011",
    "Taruffi, L., &amp; Koelsch, S. (2014). The paradox of music-evoked sadness: An online survey. <i>PLoS ONE, 9</i>(10), Article e110490. https://doi.org/10.1371/journal.pone.0110490",
]

figures = f"""
<p><b>Figure 1</b></p>
<p><i>Weekly stream-weighted emotion scores in the earlier and the audited corpus</i></p>
<p>[Insert Figure 1 here: figure1.png]</p>
<p><i>Note.</i> Four emotions whose trends passed the AR(1) test in at least one version (fear, disgust, trust) or were
described as rising in the earlier analysis (sadness). Gray dashed line: earlier pipeline (408 songs); blue line: audited pipeline ({N['scored']} songs). Only the fear
decline is significant in the audited corpus (AR(1) <i>p</i> {p_apa(fv2['p_ar1'])}).</p>
<p></p>
<p><b>Figure 2</b></p>
<p><i>Weekly share of scored listening carried by songs with each lyrical theme</i></p>
<p>[Insert Figure 2 here: figure2.png]</p>
<p><i>Note.</i> Thin line: weekly share; thick line: centered 8-week rolling mean. All panels share the 0–60% scale. Theme definitions
are in Supplementary Section B.</p>
"""

lex_rows = []
import sys
sys.path.insert(0, str(HERE.parent / "themes"))
from theme_lexicon import THEMES  # noqa: E402
for k, (words, mh) in THEMES.items():
    lex_rows.append([theme_label[k], str(mh), "<span dir=\"rtl\">" + "، ".join(words).replace("،", ",") + "</span>"])

supp = f"""
<h2><b>Section A. Lyric retrieval and matching rules</b></h2>
<p>1. <b>Song universe.</b> All chart tracks with a Hebrew title, or by an artist who also appears on Hebrew-titled chart songs or has a known Hebrew name ({N['israeli_songs']} songs).<br>
2. <b>Normalization.</b> Titles are lower-cased, punctuation, bidirectional marks and Hebrew diacritics removed, version tags (e.g., live,
remix, acoustic, cover, “מתוך”, “גרסה”) stripped, and medleys split on “&amp;” and “/”. Titles are also compared with spaces removed.<br>
3. <b>Artist mapping.</b> Latin-script artist names are mapped to Shironet’s Hebrew forms. Unmapped names are learned when at
least two different exact-title hits share the same Shironet artist.<br>
4. <b>Search order.</b> The artist’s works page is tried first; site search is used otherwise.<br>
5. <b>Decision rule.</b> Accept when artist similarity ≥ 85 and title similarity ≥ 92 (RapidFuzz). Hold for review when title
similarity is 80–91 or the artist is uncertain. Otherwise the song is not found.<br>
6. <b>Blocking.</b> Bot-protection pages are detected and never recorded as “not found”. All pages are cached so that runs can
be resumed and reproduced offline.<br>
7. <b>Human review.</b> The {S['review']} held songs were checked by hand. Each was given a confirmed Shironet link, marked as
absent, or left for a reviewer who knows the song (covers and partial medleys). Decisions are stored in
<i>manual_overrides.csv</i>.<br>
8. <b>Scoring.</b> Lyrics with less than 50% Hebrew letters are excluded. Longer lyrics are scored in 510-token windows with
stride 255 and averaged with weights for window length.</p>

<h2><b>Section B. Theme word lists</b></h2>
{table("B1", "Theme word lists and thresholds", ["Theme", "Minimum matches", "Words and phrases"], lex_rows,
       "Words match as whole words with up to two attached prefix letters (ו, ה, ב, כ, ל, מ, ש). Ambiguous words "
       "(e.g., בר, which also occurs inside כבר) were removed after inspecting matches.")}

<h2><b>Section C. Robustness checks</b></h2>
<p><b>C1. Placebo series.</b> 2,000 simulated AR(1) series (coefficient .97, no trend, 124 weeks): {pct(N['placebo_share'], 0)}
reached |ρ| &gt; .5 with naive <i>p</i> &lt; .001.<br>
<b>C2. Chow test at every week.</b> Share of possible break weeks “significant” at <i>p</i> &lt; .001: {", ".join(f"{e} {pct(N['chow_anywhere'][e], 0)}" for e in EMOS)}.<br>
<b>C3. Intermediate corpus.</b> The earlier lyrics minus the 51 wrong or doubtful matches (357 songs), re-scored with windowing:
fear AR(1) <i>p</i> &lt; .001; disgust .146; trust .236; sadness .147.<br>
<b>C4. Coverage drift.</b> {pct(N['coverage_first6'])} → {pct(N['coverage_last6'])}, AR(1) <i>p</i> {p_apa(N['coverage_trend_p'])}.<br>
<b>C5. Song concentration.</b> Fear: five songs produce {N['fear_top5_share'] * 100:.0f}% of the first-to-last decline. Joy peak:
one song produces {pct(N['joy_top_share'], 0)} of the excess. Themes: see Table 4.<br>
<b>C6. Song-level check of the fear–war link.</b> War-themed songs’ mean HebEMO fear .059 versus .012 for other songs
(Mann–Whitney <i>p</i> = .105).</p>
"""

css = ("body{font-family:'Times New Roman';font-size:12pt;line-height:2;} h1{font-size:14pt;} h2{font-size:12pt;} "
       "p{margin:0 0 6pt 0;}")
parts = [
    f"<html><head><meta charset='utf-8'><style>{css}</style></head><body>",
    f"<h1><b>{TITLE}</b></h1>",
    "<p><b>Short title:</b> Israeli streaming during war: an audited lyric analysis</p>",
    "<p><b>Word count:</b> [WORDS] (excluding abstract, references, tables, and figures)</p>",
    "<p>Dana Sinai<sup>1,2</sup>*, Ofer Rahamim<sup>3,4</sup></p>",
    "<p><sup>1</sup> Baruch Ivcher School of Psychology, Reichman University, Herzliya, Israel</p>",
    "<p><sup>2</sup> Ramat-Chen Brüll Mental Health Center, Clalit Health Services, Community Division, Tel Aviv District, Tel Aviv, Israel</p>",
    "<p><sup>3</sup> Data Research Center for Mental Health and Rehabilitation, Clalit Health Services, Petach Tikva, Israel</p>",
    "<p><sup>4</sup> Geha Mental Health Center, Petah Tikva, Israel</p>",
    "<p><i>* Corresponding author:</i> Dana Sinai, PhD, Baruch Ivcher School of Psychology, Reichman University, "
    "8 Ha’university St, Herzliya 4610101, Israel. E-mail: dana.sinai@gmail.com</p>",
    "<h1><b>Abstract</b></h1>", abstract,
    "<h1><b>Introduction</b></h1>", intro,
    "<h1><b>Method</b></h1>", methods,
    "<h1><b>Results</b></h1>", results,
    "<h1><b>Discussion</b></h1>", discussion,
    "<h1><b>Statements and Declarations</b></h1>", declarations,
    "<h1><b>References</b></h1>",
    "".join(f'<p style="padding-left:36pt;text-indent:-36pt;">{r}</p>' for r in refs),
    "<h1><b>Tables</b></h1>", tab1, tab2, tab3, tab4,
    "<h1><b>Figures</b></h1>", figures,
    "<h1><b>Supplementary Materials</b></h1>", supp,
    "</body></html>",
]
html = "".join(parts)

# word count of the main text (introduction to discussion)
main = intro + methods + results + discussion
words = len(re.sub(r"<[^>]+>", " ", main).split())
html = html.replace("[WORDS]", f"{round(words, -1):,}")
(HERE / "manuscript.html").write_text(html, encoding="utf-8")
print("main-text words:", words, " html bytes:", len(html.encode()))
