"""Candidate scoring with the artist as a hard gate.

Decisions:
  accept  - title matches AND a credited performer matches the Shironet artist
            (or the Shironet page belongs to an artist ID already confirmed for this
            Spotify artist).
  review  - title matches but the artist does not (likely a cover or a different song
            with the same name), or the title is only a near match. Never used
            automatically; goes to the manual review queue.
  reject  - everything else.
The old rule (0.8*title + 0.2*artist > 70) accepted any exact-title hit even with
artist similarity 0, which put 43 wrong songs into the corpus.
"""
from __future__ import annotations

from rapidfuzz import fuzz

from normalize import NON_PERFORMERS, clean, squash, split_artists

TITLE_ACCEPT = 92   # on the whitespace-free form
TITLE_REVIEW = 80
ARTIST_ACCEPT = 85


def title_score(spotify_title: str, shironet_title: str) -> float:
    a, b = squash(spotify_title), squash(shironet_title)
    if not a or not b:
        return 0.0
    # token_sort handles word order ('פרופיל 97' vs '97 פרופיל'); ratio on the squashed
    # form handles spelling variants ('אייכה' vs 'איכה') and the site's glued words
    return max(fuzz.ratio(a, b), fuzz.token_sort_ratio(clean(spotify_title), clean(shironet_title)))


def artist_variants(spotify_artists: str, mapping: dict) -> list[str]:
    out = []
    for a in split_artists(spotify_artists):
        if a in NON_PERFORMERS:
            continue
        out.append(a)
        heb = mapping.get(a)
        if isinstance(heb, str):
            out += [h.strip() for h in heb.split(",") if h.strip()]
    whole = mapping.get(spotify_artists)
    if isinstance(whole, str):
        out += [h.strip() for h in whole.split(",") if h.strip()]
    return list(dict.fromkeys(v for v in out if v))


def artist_score(variants: list[str], shironet_artist: str) -> float:
    s = clean(shironet_artist)
    if not s or not variants:
        return 0.0
    best = 0.0
    for v in variants:
        v = clean(v)
        if not v:
            continue
        # token_set handles "טונה מארח את נצ'י נצ'" vs "טונה"; ratio guards against short-string partials
        sc = max(fuzz.token_set_ratio(v, s), fuzz.ratio(v, s))
        if len(v) >= 4:
            sc = max(sc, fuzz.partial_ratio(v, s))
        best = max(best, sc)
    return best


def decide(spotify_title: str, variants: list[str], cand: dict, known_prfids: set[str]) -> dict:
    ts = title_score(spotify_title, cand["title"])
    ars = artist_score(variants, cand.get("artist", ""))
    prfid_ok = cand.get("prfid") in known_prfids
    artist_ok = ars >= ARTIST_ACCEPT or prfid_ok
    if ts >= TITLE_ACCEPT and artist_ok:
        d = "accept"
    elif ts >= TITLE_ACCEPT or (ts >= TITLE_REVIEW and artist_ok):
        d = "review"
    else:
        d = "reject"
    return {**cand, "title_score": round(ts, 1), "artist_score": round(ars, 1),
            "prfid_known": prfid_ok, "decision": d}


def learn_artist_names(hits: dict[str, list[tuple[str, str, str]]], min_songs: int = 2) -> dict[str, set[str]]:
    """Infer Shironet artist IDs for Spotify artists missing from the Hebrew-name table.

    hits: spotify_artist -> [(song_key, shironet_artist, prfid)] for exact-title (>=TITLE_ACCEPT)
    search hits. If the same Shironet artist appears for >= min_songs *different* songs of
    that Spotify artist, it is taken to be them (e.g. 'Noam Bettan' -> 'נועם בתן').
    One shared title can be coincidence; two different titles by the same artist is not.
    """
    learned: dict[str, set[str]] = {}
    for sp_artist, rows in hits.items():
        by_pid: dict[str, set[str]] = {}
        for song, _name, pid in rows:
            if pid:
                by_pid.setdefault(pid, set()).add(song)
        ok = {pid for pid, s in by_pid.items() if len(s) >= min_songs}
        if ok:
            learned[sp_artist] = ok
    return learned


def best(decided: list[dict]) -> dict | None:
    rank = {"accept": 2, "review": 1, "reject": 0}
    if not decided:
        return None
    return max(decided, key=lambda c: (rank[c["decision"]], c["title_score"] + c["artist_score"]))
