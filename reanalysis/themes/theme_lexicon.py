"""
Transparent theme coding of Hebrew lyrics (exploratory, complements HebEMO).

Each theme is a list of whole Hebrew word forms (and short phrases), matched as complete
words with up to two attached prefix letters (ו ה ב כ ל מ ש). A song carries a theme if it has >= MIN_HITS
matches (war/grief need 1 because their vocabulary is specific and rare in pop).
Outputs per-song theme flags and weekly stream-weighted theme shares.

Usage: PIPELINE_OUT=lyrics_pipeline/out_mac/out python reanalysis/themes/theme_lexicon.py
"""
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("PIPELINE_OUT", ROOT / "lyrics_pipeline/out"))
CHARTS = ROOT / "lyrics_pipeline/data/spotify_israel_combined.csv"
RES = Path(__file__).parent

THEMES = {
    # whole word forms (with optional attached prefixes); ambiguous everyday words deliberately left out
    "war_security": (["מלחמה", "מלחמות", "המלחמה", "חייל", "חיילת", "חיילים", "חיילות", "צבא", "הצבא", "צה\"ל",
                      "לוחם", "לוחמת", "לוחמים", "מדים", "טיל", "טילים", "אזעקה", "אזעקות", "מקלט", "ממ\"ד",
                      "קרב", "קרבות", "נשק", "רובה", "מילואים", "מילואימניק", "פצוע", "פצועים", "חטוף", "חטופה",
                      "חטופים", "בשבי", "עזה", "לבנון", "יחידה", "שכול", "שכולה", "הרוג", "הרוגים", "חללים",
                      "ניצחון", "אויב", "אויבים", "חזית", "פיקוד", "מחבלים", "מחבל", "שבעה באוקטובר"], 1),
    "grief_loss": (["קבר", "קברים", "לוויה", "יזכור", "שכול", "שכולה", "יתום", "יתומה", "אלמנה", "נרצח", "נרצחה",
                    "נרצחו", "נפל בקרב", "נפלו", "הנופלים", "מצבה", "בית קברות", "לזכרו", "לזכרה"], 1),
    "faith_prayer": (["אלוהים", "אלוקים", "ה'", "השם", "הקב\"ה", "בורא", "תפילה", "תפילות", "מתפלל", "מתפללת",
                      "מתפללים", "להתפלל", "אמונה", "ריבונו", "יתברך", "גאולה", "ישועה", "אמן", "בס\"ד",
                      "תהילים", "אבא שבשמיים", "בשם השם", "רבונו", "קדוש", "שכינה"], 2),
    "hope_resilience": (["יהיה טוב", "תקווה", "ננצח", "נתגבר", "ביחד", "נחזור", "יחזרו", "לטובה", "נעבור",
                         "חזקים", "לא נוותר", "נשרוד", "נקום", "הכל לטובה", "עוד יהיה", "נפרח"], 2),
    "nation_home": (["ארץ ישראל", "עם ישראל", "מדינה", "המדינה", "ירושלים", "מולדת", "מולדתי", "דגל", "הדגל",
                     "ישראל", "הארץ הזאת", "ארצי", "עמי", "ציון"], 2),
    "romance_heartbreak": (["אהבה", "האהבה", "אהבתי", "אהבת", "אוהב", "אוהבת", "מאמי", "נשיקה", "נשיקות", "פרידה",
                            "נפרדנו", "עזבת", "עזבתי", "הלכת", "בגדת", "בוגד", "בוגדת", "נשבר", "נשברתי", "נשברת",
                            "שבור", "שבורה", "הלב שלי", "ליבי", "מתגעגע", "מתגעגעת", "געגוע", "געגועים", "אקס"], 3),
    "party_hedonism": (["מסיבה", "מסיבות", "לרקוד", "רוקד", "רוקדת", "רוקדים", "שותה", "שותים", "לשתות",
                        "שתיתי", "משתכר", "משתכרת", "שיכור", "שיכורה", "טקילה", "וודקה", "ערק", "רום",
                        "מועדון", "וויסקי", "ויסקי", "שמפניה", "צ'ייסרים", "צ׳ייסר", "בקבוק"], 2),
}
MIN_HITS = {k: v[1] for k, v in THEMES.items()}
PREFIX = "[והבכלמש]{0,2}"


def pattern(words):
    alts = "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))
    return re.compile(rf"(?<![֐-׿]){PREFIX}(?:{alts})(?![֐-׿])")


PATTERNS = {k: pattern(v[0]) for k, v in THEMES.items()}


def code_song(text: str) -> dict:
    out = {}
    for k, p in PATTERNS.items():
        n = len(p.findall(text))
        out[f"{k}_hits"] = n
        out[k] = int(n >= MIN_HITS[k])
    return out


def main():
    m = pd.read_csv(OUT / "matches.csv")
    m = m[m.lyrics_file.notna()]
    rows = []
    for r in m.itertuples():
        t = (OUT / "lyrics" / r.lyrics_file).read_text(encoding="utf-8")
        rows.append({"uri": r.uri, "track_name": r.track_name, "artist_names": r.artist_names, **code_song(t)})
    songs = pd.DataFrame(rows)
    songs.drop(columns=["track_name", "artist_names"]).to_csv(RES / "song_themes.csv", index=False)
    themes = list(THEMES)
    c = pd.read_csv(CHARTS, parse_dates=["week_date"]).merge(songs, on="uri")
    w = c.groupby("week_date").apply(lambda x: pd.Series({t: np.average(x[t], weights=x.streams) for t in themes}))
    w.to_csv(RES / "weekly_theme_shares.csv")
    print("share of songs with each theme:\n" + songs[themes].mean().round(3).to_string())
    print("\nmean weekly stream share:\n" + w.mean().round(3).to_string())


if __name__ == "__main__":
    main()
