"""Polite, cached Shironet client and HTML parsers.

Every fetched page is cached on disk, so re-running the matcher (e.g. after changing
thresholds) does not hit the site again. Bot-challenge / empty pages are detected
and retried with backoff instead of being recorded as "no results" (the main
failure mode of the 2026-02 run: 164 searches came back empty).
"""
from __future__ import annotations

import hashlib
import random
import re
import time
from pathlib import Path
from urllib.parse import parse_qs, quote, urljoin, urlparse

from bs4 import BeautifulSoup

BASE = "https://shironet.mako.co.il"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept-Language": "he-IL,he;q=0.9,en;q=0.7",
}


class BlockedError(RuntimeError):
    pass


def _is_real_page(html: str) -> bool:
    # Genuine pages carry the site's search form; challenge/interstitial pages do not.
    return 'id="headerSearchForm0"' in html and "</html>" in html.lower()


class Client:
    def __init__(self, cache_dir: Path, delay: float = 2.5, max_retries: int = 5, offline: bool = False):
        import requests
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.delay, self.max_retries, self.offline = delay, max_retries, offline
        self._last = 0.0

    def _path(self, url: str) -> Path:
        return self.cache / (hashlib.sha1(url.encode()).hexdigest() + ".html")

    def get(self, url: str) -> str:
        p = self._path(url)
        if p.exists():
            return p.read_text(encoding="utf-8")
        if self.offline:
            raise FileNotFoundError(f"not cached (offline mode): {url}")
        wait = self.delay
        for attempt in range(self.max_retries):
            time.sleep(max(0.0, self._last + self.delay - time.time()) + random.uniform(0, 0.8))
            self._last = time.time()
            try:
                r = self.session.get(url, timeout=20)
                r.encoding = "utf-8"
                if r.status_code == 200 and _is_real_page(r.text):
                    p.write_text(r.text, encoding="utf-8")
                    return r.text
            except Exception:  # network hiccup -> retry
                pass
            time.sleep(wait)
            wait *= 2
        raise BlockedError(f"no valid page after {self.max_retries} tries: {url}")

    def search(self, q: str) -> str:
        return self.get(f"{BASE}/search?q={quote(q)}")

    def artist_works(self, prfid: str) -> str:
        return self.get(f"{BASE}/artist?type=works&lang=1&prfid={prfid}")


def _ids(href: str) -> dict:
    qs = parse_qs(urlparse(href).query)
    return {k: v[0] for k, v in qs.items()}


def parse_search(html: str) -> list[dict]:
    """Song hits from a /search page: title (with spaces kept), artist, prfid, wrkid, url."""
    soup = BeautifulSoup(html, "html.parser")
    out, seen = [], set()
    for a in soup.find_all("a", class_="search_link_name_big", href=True):
        h = a["href"]
        if "type=lyrics" not in h or "wrkid=" not in h or "play=true" in h:
            continue
        ids = _ids(h)
        key = (ids.get("prfid"), ids.get("wrkid"))
        if key in seen:
            continue
        seen.add(key)
        artist = ""
        td = a.find_parent("td")
        if td:
            al = td.find("a", class_="search_link_name_big",
                         href=lambda x: x and "prfid=" in x and "wrkid" not in x)
            artist = al.get_text(" ", strip=True) if al else ""
        out.append({
            # get_text(" ") keeps word boundaries; the old scraper's strip=True glued words together
            "title": re.sub(r"\s+", " ", a.get_text(" ", strip=True)),
            "artist": artist,
            "prfid": ids.get("prfid"),
            "wrkid": ids.get("wrkid"),
            "url": urljoin(BASE, h),
        })
    return out


def parse_artist_works(html: str, prfid: str) -> list[dict]:
    """All songs listed on an artist's 'works' page (fallback when search misses)."""
    soup = BeautifulSoup(html, "html.parser")
    name_el = soup.find(class_="artist_singer_title")
    artist = name_el.get_text(" ", strip=True) if name_el else ""
    out, seen = [], set()
    for a in soup.find_all("a", href=True):
        h = a["href"]
        if "type=lyrics" in h and "wrkid=" in h and f"prfid={prfid}" in h and "play=true" not in h:
            ids = _ids(h)
            title = a.get_text(" ", strip=True)
            if not title or ids.get("wrkid") in seen:
                continue
            seen.add(ids.get("wrkid"))
            out.append({"title": title, "artist": artist, "prfid": prfid,
                        "wrkid": ids.get("wrkid"), "url": urljoin(BASE, h)})
    return out


def parse_lyrics_page(html: str) -> dict | None:
    soup = BeautifulSoup(html, "html.parser")
    span = soup.find("span", class_="artist_lyrics_text")
    if not span:
        return None
    for br in span.find_all("br"):
        br.replace_with("\n")
    text = re.sub(r"\n{3,}", "\n\n", span.get_text()).strip()
    t = soup.find(class_="artist_song_name_txt")
    a = soup.find(class_="artist_singer_title")
    return {
        "lyrics": text,
        "page_title": t.get_text(" ", strip=True) if t else "",
        "page_artist": a.get_text(" ", strip=True) if a else "",
    }
