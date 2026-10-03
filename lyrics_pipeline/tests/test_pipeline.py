import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

import match as M  # noqa: E402
from normalize import base_title, squash, title_parts  # noqa: E402
from shironet import _is_real_page, parse_lyrics_page, parse_search  # noqa: E402

FIX = HERE / "fixtures"
MAP = {"Jasmin Moallem": "יסמין מועלם", "Omer Adam": "עומר אדם", "Tuna": "טונה", "Osher Cohen": "אושר כהן",
       "Full Trunk": "פול טראנק", "Blue Pill": "הפיל הכחול"}


def test_search_parser_keeps_word_spaces():
    hits = parse_search((FIX / "search_test.html").read_text(encoding="utf-8"))
    assert len(hits) >= 3
    jm = [h for h in hits if h["artist"] == "יסמין מועלם"][0]
    assert jm["title"] == "יהיה טוב"            # old scraper produced 'יהיהטוב'
    assert jm["prfid"] == "22199" and jm["wrkid"] == "57867"


def test_lyrics_page_parser():
    page = parse_lyrics_page((FIX / "lyrics_page.html").read_text(encoding="utf-8"))
    assert page["page_title"] == "יהיה טוב"
    assert page["page_artist"] == "יסמין מועלם"
    assert page["lyrics"].startswith("יהיה טוב")


def test_real_page_detection():
    assert _is_real_page((FIX / "search_test.html").read_text(encoding="utf-8"))
    assert not _is_real_page("<html><body>Please verify you are human</body></html>")


def test_title_normalization():
    assert base_title("סהרה - Live") == "סהרה"
    assert base_title("אם את הולכת - אקוסטי") == "אם את הולכת"
    assert base_title("שושנים עצובות (ישראל בידור)") == "שושנים עצובות"
    assert base_title("Metoh Mahshavot | מתוך מחשבות") == "מתוך מחשבות"
    assert base_title("טאטע תטהר - גרסה אקוסטית") == "טאטע תטהר"
    assert title_parts("יהיה טוב & מים שקופים") == ["יהיה טוב", "מים שקופים"]
    assert title_parts("רבות הדרכים / עושה לי צרות - Live") == ["רבות הדרכים", "עושה לי צרות"]
    assert squash("עזבת ת׳בית") == squash("עזבת ת'בית")


def _decide(title, artists, shiro_title, shiro_artist, prfids=frozenset()):
    v = M.artist_variants(artists, MAP)
    return M.decide(title, v, {"title": shiro_title, "artist": shiro_artist, "prfid": "1", "url": ""}, set(prfids))["decision"]


def test_wrong_artist_is_not_accepted():
    # Real cases the old rule accepted
    assert _decide("רוזה", "Omer Adam", "רוזה", "יהורם גאון") == "review"
    assert _decide("השם ירחם", "Tuna", "השם ירחם", "איתי זבולון") == "review"
    assert _decide("אהבה", "Osher Cohen", "אהבה", "ריין סובוטקה") == "review"


def test_right_artist_is_accepted():
    assert _decide("יהיה טוב", "Jasmin Moallem", "יהיה טוב", "יסמין מועלם") == "accept"
    assert _decide("עולם משוגע", "Tuna, Ravid Plotnik", "עולם משוגע", "טונה מארח את נצ'י נצ'") == "accept"
    assert _decide("הכל יהיה", "Blue Pill", "הכל יהיה", "הפיל הכחול") == "accept"
    assert _decide("וואלק", "Full Trunk", "וואלק", "Full Trunk") == "accept"
    # A confirmed Shironet artist ID overrides a spelling mismatch
    assert _decide("רוזה", "Omer Adam", "רוזה", "עמר אדם ושות'", prfids={"1"}) == "accept"


def test_cover_shows_do_not_count_as_performers():
    v = M.artist_variants("הכוכב הבא, Noam Bettan", MAP)
    assert "הכוכב הבא" not in v


def test_title_word_order_and_spelling():
    assert M.title_score("פרופיל 97", "97 פרופיל") >= M.TITLE_ACCEPT
    assert M.title_score("יהיה טוב", "יהיהטוב") == 100


def test_learn_unmapped_artist_needs_two_distinct_songs():
    hits = {"Noam Bettan": [("s1", "נועם בתן", "777"), ("s2", "נועם בתן", "777"), ("s3", "עומר אדם", "5")],
            "Someone": [("s1", "זמר אחר", "9")]}
    learned = M.learn_artist_names(hits)
    assert learned == {"Noam Bettan": {"777"}}


def test_windows_cover_long_songs_without_truncation():
    from score_hebemo import clean_lyrics, windows
    w = windows(list(range(1200)))
    assert w[0][0] == 0 and w[-1][-1] == 1199 and all(len(x) <= 510 for x in w)
    assert windows(list(range(100))) == [list(range(100))]
    assert clean_lyrics("פזמון:\nשורה אחת x2\n[...]") == "שורה אחת"
