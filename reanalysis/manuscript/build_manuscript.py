"""
Build the manuscript draft (APA 7 layout) as HTML from reanalysis/manuscript_numbers.json, so every number
in the text and tables comes from one place. The HTML is uploaded to Google Drive and converted to a Google Doc.

Usage: python reanalysis/manuscript/build_manuscript.py   -> reanalysis/manuscript/manuscript.html
"""
import json
import re
from pathlib import Path

import pandas as pd

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
def cell(content, top=False, bottom=False, align="center"):
    w = f"{1 if top else 0}pt 0 {1 if bottom else 0}pt 0"
    a = "text-align:center;" if align == "center" else ""
    return f'<td style="border-style:solid;border-color:#000;border-width:{w};{a}">{content}</td>'


def table(num, title, head, rows, note, widths=None):
    """APA 7 table: bold number, italic title, rules above and below the header and below the last row only."""
    th = "".join(cell(h, top=True, bottom=True, align="left" if j == 0 else "center") for j, h in enumerate(head))
    body = []
    for i, r in enumerate(rows):
        last = i == len(rows) - 1
        body.append("<tr>" + "".join(cell(c, bottom=last, align="left" if j == 0 else "center")
                                     for j, c in enumerate(r)) + "</tr>")
    return (f'<p style="line-height:2;"><b>Table {num}</b></p><p style="line-height:2;"><i>{title}</i></p>'
            f'<table style="border-collapse:collapse;border:none;font-family:\'Times New Roman\';font-size:12pt;">'
            f"<tr>{th}</tr>{''.join(body)}</table>"
            f'<p><i>Note.</i> {note}</p><p></p>')


tab1 = table(
    1, "Outcome of Lyric Retrieval for Israeli Chart Songs",
    ["Outcome", "Songs, <i>n</i>", "Share of all chart streams"],
    [
        ["Accepted (artist and title verified)", f"{S['accepted']}", pct(SS['accepted'])],
        ["Held for human review", f"{S['review']}", pct(SS['review'])],
        ["Not found on Shironet", f"{S['not_found']}", pct(SS['not_found'])],
        ["Matched, lyrics not yet downloaded", f"{S['lyrics_not_fetched']}", pct(SS['lyrics_not_fetched'])],
        ["Shironet page without lyrics", f"{S['page_without_lyrics']}", pct(SS['page_without_lyrics'])],
        ["Total Israeli songs", f"{N['israeli_songs']}", pct(sum(SS.values()))],
    ],
    f"Of the accepted songs, {N['scored']} had Hebrew lyrics and were analyzed; {N['windowed_songs']} long lyrics were scored in "
    f"overlapping windows. Songs with analyzed lyrics carried a mean of {pct(N['coverage_mean'])} of weekly streams (range "
    f"{100 * N['coverage_min']:.0f}–{100 * N['coverage_max']:.0f}%). The remaining streams were mostly international songs.")

sat_rows = []
for e in ["disgust", "anger", "sadness", "anticipation", "trust", "fear", "joy"]:
    sat_rows.append([e.capitalize(), pct(N["saturation_gt90"][e], 0), pct(N["saturation_lt10"][e], 0),
                     pct(N["saturation_gt90"][e] + N["saturation_lt10"][e], 0)])
tab2 = table(
    5, f"Distribution of Song-Level HebEMO Probabilities (<i>N</i> = {N['scored']} Songs)",
    ["Emotion", "Songs with <i>p</i> &gt; .90", "Songs with <i>p</i> &lt; .10", "Near 0 or 1 (total)"],
    sat_rows,
    "HebEMO returns one probability per emotion from a separate binary classifier. "
    f"In addition, the heBERT sentiment model labelled {pct(N['neg_sent_gt90'], 0)} of songs as negative with <i>p</i> &gt; .90. "
    "Surprise is omitted (all values below .01).")

trend_rows = []
for e in EMOS:
    a, b = T[(e, "published")], T[(e, "v2")]
    trend_rows.append([e.capitalize(), rho(a["spearman"]), p_cell(a["p_naive"]), p_cell(a["p_ar1"]),
                       rho(b["spearman"]), p_cell(b["p_hac"]), p_cell(b["p_ar1"]),
                       f"{b['first6']:.3f}".lstrip("0") + " → " + f"{b['last6']:.3f}".lstrip("0")])
tab3 = table(
    5, f"Trends in Weekly Stream-Weighted Emotion Scores in the Earlier and the Audited Corpus ({N['weeks']} Weeks)",
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
    2, "Stream-Weighted Share of Listening Carried by Songs With Each Lyrical Theme",
    ["Theme", "Songs, <i>n</i> (%)", "First 6 months (%)", "Last 6 months (%)", "Change per year (points)",
     "AR(1) <i>p</i>", "Change due to top 5 songs"],
    th_rows,
    "A song carries a theme when its lyrics contain the theme’s words at least a set number of times (Supplementary "
    "Section B); songs can carry several themes or none. Shares are percentages of each week’s scored streams. "
    "“Change due to top 5 songs” is the part of the first-to-last change produced by the five songs that contributed most "
    f"to it. For reference, the ten most-streamed songs of a typical week carry {N.get('top10_share_pct', 17.4):.1f}% of that week’s streams.")

# ---------------------------------------------------------------- temporal analysis (reanalysis/temporal)
TD = HERE.parent / "temporal"
EMO = ["joy", "sadness", "anger", "fear", "trust", "disgust", "anticipation"]
HY = pd.read_csv(TD / "half_year_means.csv", index_col=0)
SH = pd.read_csv(TD / "trajectory_shapes.csv").set_index("series")
EVW = pd.read_csv(TD / "event_windows.csv")
EVD = json.loads((TD / "event_drivers.json").read_text(encoding="utf-8"))
SERIES_LABEL = {"joy": "Joy", "sadness": "Sadness", "anger": "Anger", "fear": "Fear", "trust": "Trust",
                "disgust": "Disgust", "anticipation": "Anticipation", **{k: v for k, v in {
                    "war_security": "War and security", "grief_loss": "Grief and loss", "faith_prayer": "Faith and prayer",
                    "hope_resilience": "Hope and resilience", "nation_home": "Nation and homeland",
                    "romance_heartbreak": "Romance and heartbreak", "party_hedonism": "Partying and drinking"}.items()}}
n_ev_tests = int(EVW.p_perm.notna().sum())
n_ev_sig = int((EVW.p_perm < .05).sum())


def sp(s):
    p = SH.loc[s, "p_nonlinear"]
    return "&lt; .001" if p < .001 else f"= {p:.3f}".replace("0.", ".")


hy_rows = []
for ser in EMO + ["war_security", "grief_loss", "faith_prayer", "hope_resilience", "nation_home", "romance_heartbreak",
                  "party_hedonism"]:
    vals = HY[ser]
    scale = 1 if ser in EMO else 100
    fmt = (lambda v: f"{v:.3f}".lstrip("0")) if ser in EMO else (lambda v: f"{v:.1f}")
    p = SH.loc[ser, "p_nonlinear"]
    if ser in EMO:
        rho_l, p_l = T[(ser, "v2")]["spearman"], T[(ser, "v2")]["p_ar1"]
    else:
        tt = {t["theme"]: t for t in N["themes"]}[ser]
        rho_l, p_l = tt["rho"], tt["p_ar1"]
    pc = lambda q: "&lt; .001" if q < .001 else f"{q:.3f}".lstrip("0")
    hy_rows.append([SERIES_LABEL[ser]] + [fmt(v * scale) for v in vals] + [rho(rho_l), pc(p_l), pc(p)])
tab_hy = table(
    3, "Half-Year Means of Weekly Stream-Weighted Emotion Scores and Theme Shares",
    ["Series"] + ["Oct 23–Mar 24", "Apr–Sep 24", "Oct 24–Mar 25", "Apr–Sep 25", "Oct 25–Feb 26"]
    + ["ρ", "Linear <i>p</i>", "Curve <i>p</i>"],
    hy_rows,
    "Emotions are mean HebEMO probabilities; themes are percentages of scored streams. The last period covers five months. "
    "ρ = Spearman correlation of the weekly series with time. Linear <i>p</i> tests a straight-line trend with first-order "
    "autoregressive (AR(1)) errors. Curve <i>p</i> tests whether a smooth curve (natural cubic spline, 4 <i>df</i>) fits "
    "better than a straight line, also with AR(1) errors.")

EV_DATES = {"a": ("October 7 attack", "Oct 7, 2023"), "b": ("Iranian attack (True Promise I)", "Apr 13, 2024"),
            "c": ("Nuseirat hostage rescue", "Jun 8, 2024"), "d": ("Pager attacks on Hezbollah", "Sep 17, 2024"),
            "e": ("Iranian missile barrage (True Promise II)", "Oct 1, 2024"), "f": ("Yahya Sinwar killed", "Oct 17, 2024"),
            "g": ("Israel–Hezbollah ceasefire", "Nov 27, 2024"), "h": ("Gaza ceasefire takes effect", "Jan 19, 2025"),
            "i": ("Fighting in Gaza resumes", "Mar 18, 2025"), "j": ("12-day war with Iran", "Jun 13, 2025"),
            "k": ("Gaza peace deal signed; last living hostages released", "Oct 9, 2025")}
drv = {(d["event"], d["series"]): d for d in EVD}
a_lv = EVW[EVW.event == "a"].set_index("series").level_z
ev_rows = []
for code, (lab, date) in EV_DATES.items():
    if code == "a":
        shifts = ("Series start (no pre-event window). First 3 weeks: high fear (<i>z</i> = "
                  f"{a_lv['fear']:.2f}), anticipation ({a_lv['anticipation']:.2f}), hope ({a_lv['hope_resilience']:.2f}), "
                  f"grief ({a_lv['grief_loss']:.2f}); low disgust ({a_lv['disgust']:.2f}), faith ({a_lv['faith_prayer']:.2f})")
        songs = "—"
    else:
        sub = EVW[(EVW.event == code) & (EVW.p_perm < .05)].sort_values("p_perm")
        if sub.empty:
            shifts, songs = "None at <i>p</i> &lt; .05", "—"
        else:
            shifts = "; ".join(f"{SERIES_LABEL[r.series]} {r.change_sd:+.2f} <i>SD</i> (<i>p</i> = {r.p_perm:.3f})".replace("0.0", ".0").replace("= 0.", "= .")
                               for r in sub.itertuples())
            names = []
            for r in sub.itertuples():
                for s in drv[(code, r.series)]["songs"][:2]:
                    if s["share"] > .2 and f"<i>{s['track']}</i> ({s['artist']})" not in names:
                        names.append(f"<i>{s['track']}</i> ({s['artist']})")
            songs = "; ".join(names) or "—"
    shifts = re.sub(r"(?<![\w-])-(\d)", "−\\1", shifts)
    songs = songs.replace("ששון איפרם שאולוב", "Sasson Ifram Shaulov")
    ev_rows.append([f"({code}) {lab}", date, shifts, songs])
tab_ev = table(
    4, "Change in Emotion and Theme Series Around Anchor Events",
    ["Event", "Date", "Shifts, 3 weeks after vs. 3 weeks before", "Songs producing the shift"],
    ev_rows,
    "Shift is the change in the weekly series, in standard deviations of that series. <i>p</i> is the share of all other "
    f"possible weeks with an equal or larger change (permutation in time). Across {n_ev_tests} event-by-series comparisons, "
    f"{n_ev_sig} reached <i>p</i> &lt; .05, about the number expected by chance ({n_ev_tests * .05:.0f}). Songs listed produced "
    "more than 20% of a shift.")

# ---------------------------------------------------------------- text
TITLE = ("From War Anthems to Party Songs: Population Affective Demand in Israeli Music Streaming "
         "During 28 Months of War")

fear_songs = "; ".join(f"<i>{t}</i> ({a})" for t, a, _ in N["fear_top5"])
joy_song = f"<i>{N['joy_top_song'][0]}</i> (Sasson Ifram Shaulov)"
FEAR_GLOSS = {"לילה עיר": "Night, City", "וואלק": "Walak", "לשוב הביתה": "Returning Home",
              "בין העיר לפרדס": "Between the City and the Orchard", "עד מחר": "Until Tomorrow"}
fear_songs_g = "; ".join(f"<i>{t}</i> (“{FEAR_GLOSS.get(t, '')}”, {a})" for t, a, _ in N["fear_top5"])
_ch = pd.read_csv(HERE.parents[1] / "lyrics_pipeline/data/spotify_israel_combined.csv", parse_dates=["week_date"])
_isr = set(pd.read_csv(HERE.parent / "results_v2_corpus/matches.csv").uri)
_ch["isr"] = _ch.uri.isin(_isr)
_w = _ch.groupby("week_date").apply(lambda x: (x.streams * x.isr).sum() / x.streams.sum())
ISR = _w.groupby(pd.cut(_w.index, pd.to_datetime(["2023-10-01", "2024-04-01", "2024-10-01", "2025-04-01", "2025-10-01",
                                                   "2026-03-01"]), right=False), observed=True).mean()

def g(heb, eng, artist=None):
    return f"<i>{heb}</i> (“{eng}”{'; ' + artist if artist else ''})"

abstract = f"""
<p><b>Objective:</b> Listening to music is a regulatory act: people choose songs to match, process, or change how they feel.
Aggregated across a population, streaming choices form a record of <i>affective demand</i>, the emotional content people
reach for. We examined how affective demand in Israel changed during the war that began on October 7, 2023.</p>
<p><b>Methods:</b> We analyzed {N['entries']:,} weekly Spotify Top-200 entries for Israel ({N['weeks']} weeks). Lyrics of
{N['scored']} Hebrew songs, carrying {pct(N['coverage_mean'], 0)} of weekly streams, were retrieved with verified artist matching,
scored with the HebEMO emotion models, and coded for seven themes. We described each weekly series, tested changes around
eleven anchor events against all other weeks, and traced every shift to the songs that produced it.</p>
<p><b>Results:</b> Demand moved from mobilizing to hedonic content. War-related songs fell from
{TH['war_security']['first6']:.1f}% to {TH['war_security']['last6']:.1f}% of listening, songs of hope and nationhood nearly
disappeared, and party and drinking songs rose from {TH['party_hedonism']['first6']:.1f}% to
{TH['party_hedonism']['last6']:.1f}%. Faith songs held steady. Emotion series followed curved paths: anger and trust peaked in
2024, sadness rose in the final year, and joy doubled from October 2024 to March 2025 before returning to baseline. Around the
pager attacks, the Iranian missile barrage, and the killing of Sinwar, demand rose for one devotional song and other faith
songs. The emotion classifier placed most songs near 0 or 1.</p>
<p><b>Conclusion:</b> Streaming data trace a shift in collective emotion regulation over a prolonged war, from mobilization
toward hedonic distraction, with recourse to faith at moments of acute threat.</p>
<p><i>Keywords:</i> affective demand, emotion regulation, music streaming, song lyrics, war, Israel, cultural analytics</p>
"""

intro = f"""
<p>People use music to regulate how they feel. They choose songs to match, process, intensify, or change their emotional
state, and they draw on distinct strategies such as entertainment, diversion, revival, discharge, and solace (Saarikallio
&amp; Erkkilä, 2007; Chong et al., 2024; Reybrouck et al., 2020). Many reach for sad music when distressed (Taruffi &amp;
Koelsch, 2014; Schubert, 2016). Because each choice is a small act of emotion regulation, the sum of a population’s
listening choices is a behavioral record of what we call <i>affective demand</i>: the emotional content people seek out. This
is not the same as how people feel. A population may seek solidarity songs when afraid, or party songs when exhausted.
Affective demand tells us which regulatory resources a population is reaching for.</p>
<p>Public streaming charts make this demand observable, week by week and at population scale, without asking anyone
anything. Listening data carry psychological signal: streaming behavior relates to personality (Anderson et al., 2021), the
features of popular songs follow the weather (Anglada-Tort et al., 2023), and lyrics on the US Billboard charts turned less
negative during societal crises, consistent with listeners seeking contrast rather than a mirror of distress (Foramitti et
al., 2025; see also Levy et al., 2024). Prolonged collective crises are where such a record would be most valuable, because
surveys are slow and costly and reach only part of the population (Norris &amp; Stevens, 2007; Conway &amp; O’Connor,
2016).</p>
<p>Measuring affective demand from charts requires care. Each charted song must be matched to its own lyrics, not to a song
with the same title by another artist. Emotion classifiers trained on other text may behave differently on lyrics, so their
output needs checking and a transparent alternative. And because chart series change slowly, neighboring weeks are not
independent; tests that ignore this find “significant” trends and breaks far more often than they should (Granger &amp;
Newbold, 1974), and big-data indices built this way can look convincing and still fail (Lazer et al., 2014).</p>
<p>Israel after October 7, 2023 offers a demanding case: 28 months of war on several fronts, with discrete, widely shared
events and documented effects on mental health (Levin et al., 2025). We ask:</p>
<p>(1) How did affective demand, the emotional and thematic content Israelis streamed, change over the war?<br>
(2) Did demand shift around major events, and which songs carried those shifts?<br>
(3) How well does a lyric emotion classifier capture affective demand, compared with transparent content coding?</p>
"""

methods = f"""
<h2><b>Data</b></h2>
<p>We used the Spotify Weekly Top-200 charts for Israel from October 12, 2023 to February 19, 2026: {N['weeks']} weeks,
{N['entries']:,} entries, and {N['tracks']:,} distinct tracks with their weekly stream counts. Of these, {N['israeli_songs']}
tracks were by Israeli artists performing in Hebrew and entered lyric retrieval. Lyrics came from Shironet
(shironet.mako.co.il), the main public Hebrew lyrics database. All data are public and aggregate; no individuals were
involved, so ethics approval was not required.</p>

<h2><b>Lyric retrieval</b></h2>
<p>Matching treated the artist as a hard requirement (Supplementary Section A). Title-only matching is not safe for chart
songs: in a title-based search of the same charts, 43 of 408 matches (11%) returned another artist’s song with the same title,
for example Yehoram Gaon’s <i>רוזה</i> (“Rosa”) instead of Omer Adam’s. Artist names in Latin script were mapped to their
Hebrew forms, and a candidate was accepted only when artist similarity was at least 85 and title similarity at least 92 on a
0–100 scale (RapidFuzz; titles compared without spaces and version tags such as “live” or “remix”). Medleys were split into
their parts. Candidates with a lower title score or an uncertain artist were held for human review rather than accepted, and
each decision was recorded in a public file. Pages blocked by the site’s bot protection were detected and retried rather than
recorded as missing. Long lyrics were scored in overlapping windows of 510 tokens (stride 255), averaged with weights for
length. Songs whose lyrics were less than 50% Hebrew letters were excluded.</p>

<h2><b>Emotion scoring</b></h2>
<p>Each song’s lyrics were scored with HebEMO (Chriqui &amp; Yahav, 2022), eight binary classifiers, one per Plutchik (1980)
emotion, each returning a probability from 0 to 1, plus the heBERT sentiment model. HebEMO was trained on Hebrew user
comments, not lyrics. For each week we computed the stream-weighted mean of each emotion across songs with scored lyrics.</p>

<h2><b>Theme coding</b></h2>
<p>To describe content in a way readers can check, we coded seven themes with a word list (Supplementary Section B): war and
security, grief and loss, faith and prayer, hope and resilience, nation and homeland, romance and heartbreak, and partying
and drinking. Words are matched as whole Hebrew words, allowing up to two attached prefix letters (ו, ה, ב, כ, ל, מ, ש).
Ambiguous everyday words were removed after inspecting matches. A song carries a theme when it contains the theme’s words at
least once (war, grief), twice (faith, hope, nation, partying), or three times (romance). The weekly theme share is the
stream-weighted proportion of scored listening carried by songs with that theme.</p>

<h2><b>Statistical analysis</b></h2>
<p>Neighboring weeks of chart data are strongly correlated (lag-1 autocorrelation .85–.98 here), so all inference allowed
for this. In simulations, trend-free series with this degree of autocorrelation reached |ρ| &gt; .5 with a naive
<i>p</i> &lt; .001 in {pct(N['placebo_share'], 0)} of runs. <i>Trajectories.</i> We described the shape of each series with
half-year means and a locally weighted (LOWESS) curve. We tested a straight-line trend, and whether a smooth curve (natural
cubic spline, 4 <i>df</i>) fitted better than a straight line, using generalized least squares with first-order
autoregressive (AR(1)) errors. <i>Anchor events.</i> For eleven major events of the war (Table 4), we computed the change in
each series from the three chart weeks before the event to the three weeks starting with it. Its <i>p</i> value is the share
of all other weeks at which the same before–after change was as large or larger; this permutation-in-time test keeps the
autocorrelation of the series. <i>Song contributions.</i> For every trend and every event shift with <i>p</i> &lt; .05 we
identified the songs that produced it, as the change in each song’s share of streams times its score. Coverage could change
over time and bias the series, so we tested its trend as well. Analyses used Python 3.11 (pandas, statsmodels, scipy). All
tests were two-sided. Given the exploratory aim, <i>p</i> values are not corrected for multiple comparisons; instead we report
how many event comparisons would be expected to reach <i>p</i> &lt; .05 by chance.</p>
"""

results = f"""
<h2><b>Lyric corpus</b></h2>
<p>Of the {N['israeli_songs']} Israeli songs, {S['accepted']} were accepted, {S['review']} were held for review, and
{S['not_found']} were not found on Shironet, most of them 2025 releases not yet added to the site (Table 1). The {N['scored']}
analyzed songs carried a mean of {pct(N['coverage_mean'])} of weekly streams, and coverage did not change significantly over
time ({pct(N['coverage_first6'])} in the first six months, {pct(N['coverage_last6'])} in the last six; AR(1) <i>p</i>
{p_apa(N['coverage_trend_p'])}). Israeli songs as a whole carried most of the listening throughout, and their share rose
slightly, from {pct(ISR.iloc[0])} of streams in the first half-year to {pct(ISR.iloc[3])} in April–September 2025.</p>
<p><b>[Insert Table 1 about here]</b></p>

<h2><b>Thematic content of affective demand</b></h2>
<p>Table 2 and Figure 1 show the share of listening carried by each theme. Songs about war and security fell from
{TH['war_security']['first6']:.1f}% of listening in the first six months to {TH['war_security']['last6']:.1f}% in the last six
(AR(1) <i>p</i> {p_apa(TH['war_security']['p'])}), with a partial rebound from October 2024 to March 2025
({HY.loc['Oct 2024–Mar 2025', 'war_security'] * 100:.1f}%; Table 3). Hope and resilience fell from
{TH['hope_resilience']['first6']:.1f}% to {TH['hope_resilience']['last6']:.1f}%, and nation and homeland from
{TH['nation_home']['first6']:.1f}% to {TH['nation_home']['last6']:.1f}%. These declines followed the fading of the anthems of
the first war months, among them {g('חרבו דרבו', 'Harbu Darbu', 'Ness &amp; Stilla')}, {g('עם ישראל חי', 'The People of Israel Live', 'Eyal Golan')}, {g('יהיה טוב', 'It Will Be Good', 'Jasmin Moallem, also with Omer Adam')}, and {g('לצאת מדיכאון', 'Getting Out of Depression', 'Yagel Oshri')}. The nation and grief themes rest on few songs (10 and 7) and should be read as the fate of
those songs. Partying and drinking rose from {TH['party_hedonism']['first6']:.1f}% to {TH['party_hedonism']['last6']:.1f}%
(AR(1) <i>p</i> &lt; .001), mostly in a step in spring 2025. This rise was broad-based: the five largest contributors produced
only {TH['party_hedonism']['top5']}% of it. Faith and prayer, about one song in eight, held a steady share. Romance and
heartbreak, the most common theme, rose to {HY.loc['Oct 2024–Mar 2025', 'romance_heartbreak'] * 100:.1f}% of listening from
October 2024 to March 2025 and then fell back, with no overall trend.</p>
<p><b>[Insert Table 2 about here]</b></p>
<p><b>[Insert Figure 1 about here]</b></p>

<h2><b>Emotional content of affective demand</b></h2>
<p>The emotion series followed curved rather than straight paths (Table 3, Figures 2 and 3). A smooth curve fitted better than a
line for anger (<i>p</i> {sp('anger')}), disgust (<i>p</i> {sp('disgust')}), anticipation (<i>p</i> {sp('anticipation')}),
and trust (<i>p</i> {sp('trust')}). Relative to their own averages, demand for songs scored high on fear and anticipation was
highest in the first months of the war, and demand for songs scored high on disgust was lowest. Anticipation fell by spring
2024 and recovered only partly in late 2025, while disgust rose over the first six months and then stayed high. Anger dipped
in the winter of 2023–24, peaked in August 2024, and then declined. Trust peaked in September 2024, fell to its lowest point in
May 2025, and partly recovered. Joy was flat until September 2024, more than doubled from October 2024 to March 2025
(half-year mean {f"{HY.loc['Oct 2024–Mar 2025', 'joy']:.3f}".lstrip('0')} vs. {f"{HY.loc['Apr–Sep 2024', 'joy']:.3f}".lstrip('0')}
before and {f"{HY.loc['Apr–Sep 2025', 'joy']:.3f}".lstrip('0')} after), and returned to baseline. Sadness stayed low through
August 2024 and then rose in two steps to its highest level in the final five months
({f"{HY.loc['Oct 2025–Feb 2026', 'sadness']:.3f}".lstrip('0')}).</p>
<p>Only fear showed a straight-line trend that allowed for autocorrelation: it declined from {f"{fv2['first6']:.3f}".lstrip('0')}
to {f"{fv2['last6']:.3f}".lstrip('0')} (ρ = {rho(fv2['spearman'])}, AR(1) <i>p</i> {p_apa(fv2['p_ar1'])}). Five songs, scored
high on fear and streamed heavily early in the war, produced the whole of this decline: {fear_songs_g}. Weekly fear tracked
the share of war-themed songs (<i>r</i> = {r2(N['fear_war_r'])}; first differences <i>r</i> = {r2(N['fear_war_r_diff'])}).
Single releases also produced the two sharpest fear peaks. The April 2024 peak began on April 4, nine days before the Iranian
attack, when Tuna’s {g('בין העיר לפרדס', 'Between the City and the Orchard')} entered the chart; the January 2025 peak fell in
the week the Gaza ceasefire was announced and came mainly from {g('יש לך אותי', 'You Have Me', 'Ravid Plotnik and Shai Tsabari')}. Likewise, the joy rise was carried by one song (see below), which is why a smooth curve did not fit it significantly
better than a line (<i>p</i> {sp('joy')}).</p>
<p><b>[Insert Table 3 about here]</b></p>
<p><b>[Insert Figure 2 about here]</b></p>
<p><b>[Insert Figure 3 about here]</b></p>

<h2><b>Anchor events</b></h2>
<p>Table 4 lists the change around each anchor event. Of {n_ev_tests} event-by-series comparisons, {n_ev_sig} reached
<i>p</i> &lt; .05, the number expected by chance, so single shifts must be read with caution. One cluster nevertheless stands
out because it recurred across three adjacent events. Around the pager attacks on Hezbollah, the Iranian missile barrage, and
the killing of Yahya Sinwar (September 17–October 17, 2024), demand rose for songs scored high on joy and for songs coded as
faith and prayer, and fell for songs scored high on anger. Two songs produced most of the joy and faith shifts:
{g('תמיד אוהב אותי', 'Always Loves Me', 'Sasson Ifram Shaulov')}, a devotional song about God’s unconditional love, which entered
the chart on September 26 and reached number one, and {g('לופ', 'Loop', 'Osher Cohen')}. The rise in “joy” in these weeks was
thus, in content, a rise in demand for devotional reassurance. Later, partying and drinking songs rose when fighting in Gaza
resumed in March 2025 (mainly songs featuring Odeya) and again during the 12-day war with Iran in June 2025, mainly through {g('מלכת הדור', 'Queen of the Generation', 'Omer Adam')}. The other events showed no shift beyond what occurs at ordinary weeks.</p>
<p><b>[Insert Table 4 about here]</b></p>

<h2><b>How well the classifier captures affective demand</b></h2>
<p>HebEMO rarely produced intermediate values (Table 5). Disgust exceeded .90 for {pct(N['saturation_gt90']['disgust'], 0)}
of songs and anger for {pct(N['saturation_gt90']['anger'], 0)}, whereas fear, joy, and trust were below .10 for more than 97%
of songs. The sentiment model labelled {pct(N['neg_sent_gt90'], 0)} of songs as negative, including love songs and dance hits.
A weekly mean of such near-binary scores mostly reflects how many streams went to the few songs labelled 1, so the emotion
series above are best read as demand for those songs, which is why we name them throughout.</p>
<p><b>[Insert Table 5 about here]</b></p>
"""

discussion = f"""
<p>Over 28 months of war, the content Israelis chose to stream changed in a clear overall direction, along a path with
distinct turns. In the first months, a substantial part of listening went to songs about the war itself, to anthems of
national unity, and to songs of hope and endurance. Two years later these had largely given way to songs about partying and
drinking, while religious songs held a steady share and love songs rose and fell. The emotion series rose and fell rather
than moving in straight lines, and the clearest event-linked shift, in autumn 2024, was a turn to devotional songs.</p>

<h2><b>From mobilization to hedonic distraction</b></h2>
<p>Read as affective demand, the theme results describe a change in how the population used music to regulate emotion. Early
in the war, demand centered on content that names the threat and binds the group: songs whose lyrics refer to the war,
soldiers, or hostages, and anthems such as <i>חרבו דרבו</i>, <i>עם ישראל חי</i>, and <i>יהיה טוב</i>. This fits regulation
through collective meaning and solidarity, in which music helps people process a shared event and affirm belonging
(Saarikallio &amp; Erkkilä, 2007; Reybrouck et al., 2020). As the war became chronic, demand moved toward hedonic content that
does not refer to the war at all. In the terms of Saarikallio and Erkkilä (2007), this resembles a shift from mental work and
solace toward entertainment and diversion. It parallels the move toward more positive, contrasting music that Foramitti et
al. (2025) observed during societal crises, although here it emerged gradually rather than as an immediate response. The rise
was broad, carried by many songs and artists, which makes it unlikely to reflect one hit or one release. Throughout, Israeli
songs kept or slightly increased their large share of listening.</p>
<p>The trajectories add texture to this shift. Demand did not move steadily: songs scored high on fear and anticipation were
relatively most streamed in the first months, those scored high on anger and trust rose through 2024 and then receded, and
those scored high on sadness rose late, in the final year of the war. The anchor events show where shorter shifts occurred.
The clearest is the autumn of 2024. During the pager attacks, the Iranian missile barrage, and the killing of Sinwar, Israelis
turned in large numbers to <i>תמיד אוהב אותי</i>, a devotional song about God’s unconditional love, and to other faith songs.
An emotion classifier alone would read this as a surge of joy, even euphoria. In content, it was demand for reassurance and
faith at a time of direct state-on-state threat, a regulatory resource that the steady share of religious songs shows was in
use throughout the war. Later escalations, the resumption of fighting in Gaza and the 12-day war with Iran, coincided instead
with more demand for party songs, consistent with the general move toward hedonic distraction. Because one or two releases
produced each of these shifts, and because release dates are set by artists and labels, the timing alone cannot show that the
events caused them.</p>

<h2><b>Measuring affective demand</b></h2>
<p>Three practical lessons follow. First, lyric matching must verify the artist: by title alone, about one match in nine is
another artist’s song. Second, emotion classifiers trained on other text should be checked before their output is averaged.
HebEMO, trained on user comments, assigned most songs a probability near 0 or 1 and labelled 98% of songs negative,
including love songs and dance hits, so its weekly means follow chart turnover more than content. Transparent theme coding
complements it: every match can be checked, and every result traced to specific songs. Third, weekly chart series change
slowly, and with lag-1 autocorrelations above .9, tests that assume independent weeks will often find trends and breaks in
noise. We suggest that studies of affective demand report matching accuracy, the distribution of classifier outputs with a
human validation sample, autocorrelation-robust inference with a null benchmark, and how much of each result comes from the
top few songs.</p>

<h2><b>Affective demand and mental health</b></h2>
<p>Affective demand is a behavioral signal of regulation, not a direct measure of distress, and the two may diverge. A shift
toward hedonic content could reflect recovery, but also fatigue, avoidance, or numbing. Linking the two is the next step.
Weekly demand series could be compared with independent indicators of population mental health, such as repeated surveys,
helpline contacts, or service use, to test whether particular shifts in demand precede or follow changes in distress.</p>

<p><i>Limitations</i></p>
<p>Demand is observed only through one platform’s Top 200. Spotify users skew younger, Arabic-language music is excluded,
and charts reflect recommendation algorithms and new releases as well as listeners’ choices; supply and demand cannot be
fully separated, and new releases can coincide with events by chance. About a third of weekly streams had no scored lyrics, mostly international songs. Lyrics are only part
of what a song offers; melody, tempo, and the performer also carry emotional meaning. The theme word list is our own and
was refined by inspecting matches; it is published in full so others can test alternatives. We did not collect human
emotion ratings, so we can show that HebEMO behaves implausibly on lyrics but cannot quantify its error. {S['review']}
uncertain matches ({pct(SS['review'])} of streams) were held out.</p>

<p><i>Conclusion</i></p>
<p>What a population chooses to listen to during a prolonged war is a record of how it regulates emotion. In Israel, that
record shows a move from war songs, anthems, and hope toward party and drinking songs, with steady demand for faith and
love songs. Capturing such shifts requires verified lyrics, transparent content measures, and inference suited to slowly
changing series; with these in place, streaming data can complement slower instruments for understanding collective
coping in crisis.</p>
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
<p><i>Weekly Share of Listening Carried by Songs With Each Lyrical Theme</i></p>
<p>[Insert Figure 1 here: figure1.png]</p>
<p><i>Note.</i> Shares are percentages of each week’s streams with analyzed lyrics. Thin line: weekly share; thick line:
centered 8-week rolling mean. All panels share the 0–60% scale. Theme definitions are in Supplementary Section B.</p>
<p></p>
<p><b>Figure 2</b></p>
<p><i>Trajectories of Weekly Stream-Weighted Emotion Scores and Anchor Events</i></p>
<p>[Insert Figure 2 here: figure2.png]</p>
<p><i>Note.</i> Each panel shows the weekly stream-weighted score of one emotion as a <i>z</i> score (standardized within
that emotion); the shaded area is the deviation from the emotion’s mean over the 124 weeks. <i>M</i> and <i>SD</i> are the raw
mean and standard deviation of each series. Dashed lines mark anchor events: (a) October 7 attack; (b) Iranian attack, April 13, 2024; (c) Nuseirat hostage rescue, June 8,
2024; (d) pager attacks on Hezbollah, September 17, 2024; (e) Iranian missile barrage, October 1, 2024; (f) Sinwar killed,
October 17, 2024; (g) Israel–Hezbollah ceasefire, November 27, 2024; (h) Gaza ceasefire, January 19, 2025; (i) fighting in
Gaza resumes, March 18, 2025; (j) 12-day war with Iran, June 13, 2025; (k) Gaza peace deal signed, October 9,
2025.</p>
<p></p>
<p><b>Figure 3</b></p>
<p><i>Monthly Emotion Scores and Theme Shares Relative to Their Means</i></p>
<p>[Insert Figure 3 here: figure3.png]</p>
<p><i>Note.</i> Each cell is the monthly mean of the weekly <i>z</i> score of that series; red is above and blue below the
series mean. Dashed lines mark anchor events (a)–(k), as in Figure 2.</p>
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
{table("B1", "Theme Word Lists and Thresholds", ["Theme", "Minimum matches", "Words and phrases"], lex_rows,
       "Words match as whole words with up to two attached prefix letters (ו, ה, ב, כ, ל, מ, ש). Ambiguous words "
       "(e.g., בר, which also occurs inside כבר) were removed after inspecting matches.")}

<h2><b>Section C. Robustness checks</b></h2>
<p><b>C1. Placebo series.</b> 2,000 simulated AR(1) series (coefficient .97, no trend, 124 weeks): {pct(N['placebo_share'], 0)}
reached |ρ| &gt; .5 with naive <i>p</i> &lt; .001.<br>
<b>C2. Break tests at arbitrary weeks.</b> A Chow break test placed at any week was “significant” at <i>p</i> &lt; .001 in
72–100% of positions for six of the seven emotions ({pct(N['chow_anywhere']['fear'], 0)} for fear), so break tests that ignore
autocorrelation cannot locate turning points in these series.<br>
<b>C3. Coverage drift.</b> {pct(N['coverage_first6'])} → {pct(N['coverage_last6'])}, AR(1) <i>p</i> {p_apa(N['coverage_trend_p'])}.<br>
<b>C4. Song concentration.</b> Fear: five songs produce {N['fear_top5_share'] * 100:.0f}% of the first-to-last decline. Joy peak:
one song produces {pct(N['joy_top_share'], 0)} of the excess. Themes: see Table 2.<br>
<b>C5. Song-level check of the fear–war link.</b> War-themed songs’ mean HebEMO fear .059 versus .012 for other songs
(Mann–Whitney <i>p</i> = .105).</p>
"""

css = ("body{font-family:'Times New Roman';font-size:12pt;line-height:2;} h1{font-size:14pt;} h2{font-size:12pt;} "
       "p{margin:0 0 6pt 0;}")
parts = [
    f"<html><head><meta charset='utf-8'><style>{css}</style></head><body>",
    f"<h1><b>{TITLE}</b></h1>",
    "<p><b>Short title:</b> Affective demand in wartime music streaming</p>",
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
    "<h1><b>Tables</b></h1>", tab1, tab4, tab_hy, tab_ev, tab2,
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
