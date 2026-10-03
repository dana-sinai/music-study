"""Text normalization for matching Spotify chart entries to Shironet songs."""
import re
import unicodedata

HEB = re.compile(r"[֐-׿]")
NIQQUD = re.compile(r"[֑-ׇ]")
BIDI = re.compile(r"[‎‏‪-‮⁦-⁩]")
# Version tags that change the recording but not the lyrics
VERSION_TAGS = re.compile(
    r"\s*[-–—]\s*(live|לייב|אקוסטי|גרסה.*|גרסת.*|רמיקס|remix.*|radio edit|acoustic.*|"
    r"bonus.*|מתוך.*|sped up|slowed.*|ביצוע.*|קאבר|cover)\s*$",
    re.IGNORECASE,
)
PARENS = re.compile(r"\s*[\(\[][^\)\]]*[\)\]]")
MEDLEY_SPLIT = re.compile(r"\s+&\s+|\s+/\s+|/")
GERESH = str.maketrans({"׳": "'", "’": "'", "`": "'", "״": '"', "“": '"', "”": '"'})


def has_hebrew(s) -> bool:
    return bool(HEB.search(str(s)))


def clean(s) -> str:
    """Canonical form for comparison: no niqqud, bidi marks or punctuation; single spaces."""
    s = unicodedata.normalize("NFC", str(s))
    s = BIDI.sub("", NIQQUD.sub("", s)).translate(GERESH)
    s = re.sub(r"[^\w\s']", " ", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def squash(s) -> str:
    """Whitespace-free form. Shironet search hits lose their spaces (per-word highlight spans)."""
    return clean(s).replace(" ", "")


def base_title(title: str) -> str:
    """Strip version tags and parentheticals: 'סהרה - Live' -> 'סהרה'."""
    t = BIDI.sub("", str(title)).strip()
    if "|" in t:  # 'Metoh Mahshavot | מתוך מחשבות' -> keep the Hebrew side
        parts = [p.strip() for p in t.split("|")]
        heb = [p for p in parts if has_hebrew(p)]
        t = heb[0] if heb else parts[0]
    prev = None
    while prev != t:
        prev = t
        t = VERSION_TAGS.sub("", PARENS.sub("", t)).strip()
    return t


def title_parts(title: str) -> list[str]:
    """Split medleys: 'יהיה טוב & מים שקופים' -> ['יהיה טוב', 'מים שקופים']."""
    t = base_title(title)
    parts = [base_title(p) for p in MEDLEY_SPLIT.split(t) if p.strip()]
    return parts or [t]


def split_artists(artists: str) -> list[str]:
    return [a.strip() for a in str(artists).split(",") if a.strip()]


# Spotify "artists" that are shows/labels, not performers of the original song
NON_PERFORMERS = {"הכוכב הבא", "כאן - תאגיד השידור הישראלי", "Participants of Festigal", "Kids Choir"}
