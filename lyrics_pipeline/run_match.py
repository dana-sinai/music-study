"""Find and match Shironet lyrics for every Israeli song in the Spotify Israel weekly charts.

Usage:
  python run_match.py                 # full run (needs network access to shironet.mako.co.il)
  python run_match.py --limit 20      # first 20 songs by streams (smoke test)
  python run_match.py --offline       # recompute decisions from cached pages only

Inputs  (data/): spotify_israel_combined.csv, artist_hebrew_mapping_combined.csv,
                 manual_overrides.csv (optional; see README)
Outputs (out/):  matches.csv, candidates.jsonl, review_queue.csv, lyrics/<id>.txt
Lyrics stay in out/lyrics (git-ignored): they are copyrighted and must not be redistributed.
"""
from __future__ import annotations

import argparse
import json
from urllib.parse import parse_qs, urlparse
from collections import defaultdict
from pathlib import Path

import pandas as pd

import match as M
from normalize import NON_PERFORMERS, has_hebrew, split_artists, title_parts
from shironet import BlockedError, Client, parse_artist_works, parse_lyrics_page, parse_search

HERE = Path(__file__).parent
DATA, OUT = HERE / "data", HERE / "out"


def song_universe(charts: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Every chart track that is plausibly Israeli, not only Hebrew-titled ones."""
    g = charts.groupby("uri").agg(track_name=("track_name", "first"), artist_names=("artist_names", "first"),
                                  total_streams=("streams", "sum"), peak_rank=("rank", "min"),
                                  first_week=("week_date", "min"), weeks=("week_date", "nunique")).reset_index()
    heb_title_artists = {a for t, s in zip(charts.track_name, charts.artist_names) if has_hebrew(t)
                         for a in split_artists(s)}
    def reason(r):
        if has_hebrew(r.track_name):
            return "hebrew_title"
        arts = split_artists(r.artist_names)
        if any(has_hebrew(a) or a in mapping or a in heb_title_artists for a in arts):
            return "israeli_artist"
        return None
    g["selection"] = g.apply(reason, axis=1)
    return g[g.selection.notna()].sort_values("total_streams", ascending=False).reset_index(drop=True)


def go_offline(client: Client) -> None:
    """Shironet refused us: stop all network requests for this run (don't hammer the site),
    finish everything that can be done from cached pages, and leave the rest for a rerun."""
    if not client.offline:
        client.offline = True
        print("\n*** Shironet is blocking requests. Finishing from saved pages only; "
              "rerun later (e.g. tomorrow) to fetch the rest - saved pages are reused. ***\n", flush=True)


def seed_from_old_run(mapping: dict, known: dict) -> dict:
    """Use the Feb 2026 matches that pass the NEW rule (artist-gated) to
    (a) learn Shironet artist IDs and (b) reuse their lyrics instead of refetching.
    Returns {shironet_url: lyrics}."""
    p = DATA / "hebrew_songs_with_lyrics.csv"
    if not p.exists():
        return {}
    old = pd.read_csv(p)
    old = old[old.lyrics.notna() & old.shironet_url.notna()]
    lyr = {}
    for r in old.itertuples():
        variants = M.artist_variants(r.artist_names, mapping)
        pid = parse_qs(urlparse(r.shironet_url).query).get("prfid", [None])[0]
        c = M.decide(title_parts(r.track_name)[0], variants,
                     {"title": r.shironet_title, "artist": r.shironet_artist, "prfid": pid, "url": r.shironet_url}, set())
        if c["decision"] != "accept":
            continue
        lyr[r.shironet_url] = r.lyrics
        for a in split_artists(r.artist_names):
            if a not in NON_PERFORMERS and M.artist_score(M.artist_variants(a, mapping), r.shironet_artist) >= 90:
                known[a].add(pid)
    print(f"seeded from old run: {len(lyr)} verified lyrics, artist IDs for {len(known)} artists", flush=True)
    return lyr


def candidates_for(client: Client, part: str, variants: list[str], prfids: set[str]) -> tuple[list[dict], list[str]]:
    """Search strategies in order; stop early once an 'accept' appears."""
    cands, used = [], []
    for pid in sorted(prfids):  # artist song list first: one cached page serves all of the artist's songs
        try:
            cands += parse_artist_works(client.artist_works(pid), pid)
            used.append(f"works:{pid}")
        except FileNotFoundError:  # offline and not cached: fall through to cached searches
            pass
    if any(M.decide(part, variants, c, prfids)["decision"] == "accept" for c in cands):
        return cands, used
    queries = [part] + [f"{part} {v}" for v in variants if has_hebrew(v)][:2]
    for q in queries:
        cands += parse_search(client.search(q))
        used.append(f"search:{q}")
        if any(M.decide(part, variants, c, prfids)["decision"] == "accept" for c in cands):
            return cands, used
    return cands, used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--delay", type=float, default=6.0)
    ap.add_argument("--uri", nargs="*", help="only these Spotify URIs (testing)")
    args = ap.parse_args()

    charts = pd.read_csv(DATA / "spotify_israel_combined.csv")
    amap = pd.read_csv(DATA / "artist_hebrew_mapping_combined.csv")
    mapping = dict(zip(amap.original_artist, amap.hebrew_artist))
    ov_path = DATA / "manual_overrides.csv"
    overrides = pd.read_csv(ov_path).set_index("spotify_uri") if ov_path.exists() else pd.DataFrame()

    songs = song_universe(charts, mapping)
    if args.uri:
        songs = songs[songs.uri.isin(args.uri)]
    if args.limit:
        songs = songs.head(args.limit)
    client = Client(OUT / "cache", delay=args.delay, offline=args.offline)
    (OUT / "lyrics").mkdir(parents=True, exist_ok=True)

    # Shironet artist IDs confirmed per Spotify artist; learned in pass 1, used in pass 2
    known: dict[str, set[str]] = defaultdict(set)
    old_lyrics = seed_from_old_run(mapping, known)
    # exact-title hits for artists with no Hebrew name in the table (for M.learn_artist_names)
    unmapped_hits: dict[str, list] = defaultdict(list)
    rows, cand_log = [], []
    for pass_no in (1, 2):
        if pass_no == 2:
            for a, pids in M.learn_artist_names(unmapped_hits).items():
                known[a] |= pids
            print(f"learned Shironet IDs for {len(M.learn_artist_names(unmapped_hits))} unmapped artists", flush=True)
        rows, cand_log = [], []
        for i, s in songs.iterrows():
            arts = [a for a in split_artists(s.artist_names)]
            variants = M.artist_variants(s.artist_names, mapping)
            prfids = set().union(*(known[a] for a in arts)) if arts else set()
            row = {k: s[k] for k in ["uri", "track_name", "artist_names", "total_streams",
                                     "peak_rank", "first_week", "selection"]}
            if s.uri in overrides.index:
                o = overrides.loc[s.uri]
                row.update(status=f"override:{o.action}", shironet_urls=o.get("shironet_url", ""),
                           note=o.get("note", ""))
                rows.append(row)
                continue
            parts = title_parts(s.track_name)
            picks, status = [], "accepted"
            try:
                for part in parts:
                    cands, used = candidates_for(client, part, variants, prfids)
                    decided = [M.decide(part, variants, c, prfids) for c in cands]
                    b = M.best(decided)
                    if pass_no == 1 and not any(has_hebrew(v) for v in variants) \
                            and not any(a in NON_PERFORMERS for a in arts):
                        for c in decided:
                            if c["title_score"] >= M.TITLE_ACCEPT:
                                for a in arts:
                                    unmapped_hits[a].append((s.uri + part, c["artist"], c["prfid"]))
                    cand_log.append({"uri": s.uri, "part": part, "queries": used,
                                     "candidates": sorted(decided, key=lambda c: -c["title_score"])[:10]})
                    if b is None or b["decision"] == "reject":
                        status = "not_found"
                    elif b["decision"] == "review" and status == "accepted":
                        status = "review"
                    picks.append(b)
            except BlockedError as e:
                status, picks = "not_cached", []
                row["note"] = str(e)
                go_offline(client)
            except FileNotFoundError:
                status, picks = "not_cached", []
            row.update(status=status, n_parts=len(parts),
                       shironet_urls=" | ".join(p["url"] for p in picks if p),
                       shironet_titles=" | ".join(p["title"] for p in picks if p),
                       shironet_artists=" | ".join(p["artist"] for p in picks if p),
                       title_score=min((p["title_score"] for p in picks if p), default=None),
                       artist_score=min((p["artist_score"] for p in picks if p), default=None))
            if status == "accepted":
                for p in picks:
                    if p["artist_score"] >= 90:
                        for a in arts:
                            if M.artist_score(M.artist_variants(a, mapping), p["artist"]) >= 90:
                                known[a].add(p["prfid"])
                if pass_no == 2:
                    texts = []
                    for p in picks:
                        if p["url"] in old_lyrics:  # same Shironet page as a verified old match: no refetch
                            texts.append(old_lyrics[p["url"]])
                            continue
                        try:
                            page = parse_lyrics_page(client.get(p["url"]))
                        except BlockedError:
                            go_offline(client)
                            texts = None
                            break
                        except FileNotFoundError:  # offline after a block: fetch on the next run
                            texts = None
                            break
                        texts.append(page["lyrics"] if page else "")
                    if texts is None:
                        row["status"] = "lyrics_not_fetched"
                    elif all(texts):
                        fname = s.uri.split(":")[-1] + ".txt"
                        (OUT / "lyrics" / fname).write_text("\n\n".join(texts), encoding="utf-8")
                        row["lyrics_file"], row["lyrics_chars"] = fname, sum(map(len, texts))
                    else:
                        row["status"] = "page_without_lyrics"
            rows.append(row)
            if pass_no == 2 and (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(songs)}", flush=True)
                pd.DataFrame(rows).to_csv(OUT / "matches.partial.csv", index=False, encoding="utf-8-sig")
        print(f"pass {pass_no} done: {pd.Series([r['status'] for r in rows]).value_counts().to_dict()}", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "matches.csv", index=False, encoding="utf-8-sig")
    with open(OUT / "candidates.jsonl", "w", encoding="utf-8") as f:
        for c in cand_log:
            f.write(json.dumps(c, ensure_ascii=False, default=str) + "\n")
    q = res[~res.status.isin(["accepted"]) & ~res.status.str.startswith("override")].copy()
    q["shironet_url_correct"], q["action"], q["reviewer_note"] = "", "", ""
    q.sort_values("total_streams", ascending=False).to_csv(OUT / "review_queue.csv", index=False, encoding="utf-8-sig")

    tot = charts.streams.sum()
    share = res.groupby("status").total_streams.sum() / tot
    print("\nShare of ALL chart streams by status:\n" + share.round(3).to_string())
    print(f"\nreview queue: {len(q)} songs -> {OUT/'review_queue.csv'}")
    todo = res.status.isin(["not_cached", "lyrics_not_fetched"]).sum()
    if todo:
        print(f"\n{todo} songs still need pages from Shironet. Run the same command again later to finish them.")


if __name__ == "__main__":
    main()
