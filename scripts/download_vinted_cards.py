#!/usr/bin/env python3
"""Download 50 Vinted Pokémon single-card listing photos for CLIP tests."""
from __future__ import annotations

import http.cookiejar
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

OUT = Path("/home/nez/Projects/BattleScan/images/vinted")
UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)
API = "https://www.vinted.it/api/v2/catalog/items"
SKIP = re.compile(
    r"fan\s*art|fanart|fatta a mano|disegnata|storage box|stackable|"
    r"\blotto\b|\blot\b|set \d+ carte|3 carte|51 carte|completé vos|"
    r"aonimus|gradeado|psa\s*9|sealed",
    re.I,
)
HAS_NUM = re.compile(r"\b[a-z]{0,3}\d{1,4}\s*/\s*\d{2,4}\b", re.I)


def opener() -> urllib.request.OpenerDirector:
    jar = http.cookiejar.CookieJar()
    op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    catalog = urllib.request.Request(
        "https://www.vinted.it/catalog?search_text=pokemon&catalog_ids=4875",
        headers={"User-Agent": UA, "Accept-Language": "it-IT,it;q=0.9"},
    )
    with op.open(catalog, timeout=30) as r:
        r.read(2048)
    return op


def fetch_items(op: urllib.request.OpenerDirector, page: int) -> list[dict]:
    q = (
        f"{API}?search_text=pokemon&catalog_ids=4875&page={page}"
        f"&per_page=96&order=newest_first"
    )
    req = urllib.request.Request(
        q,
        headers={
            "User-Agent": UA,
            "Accept": "application/json",
            "Accept-Language": "it-IT,it;q=0.9",
        },
    )
    with op.open(req, timeout=30) as r:
        data = json.loads(r.read().decode())
    return data.get("items") or []


def photo_url(item: dict) -> str | None:
    photo = item.get("photo") or {}
    return photo.get("full_size_url") or photo.get("url")


def keep(item: dict) -> bool:
    title = str(item.get("title") or "")
    if SKIP.search(title):
        return False
    if not photo_url(item):
        return False
    return bool(HAS_NUM.search(title) or len(title.split()) >= 2)


def download(op: urllib.request.OpenerDirector, url: str, dest: Path) -> None:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": UA, "Referer": "https://www.vinted.it/"},
    )
    with op.open(req, timeout=45) as r:
        dest.write_bytes(r.read())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    op = opener()
    picked: list[dict] = []
    seen: set[int] = set()
    page = 1
    while len(picked) < 50 and page <= 4:
        for item in fetch_items(op, page):
            iid = int(item["id"])
            if iid in seen or not keep(item):
                continue
            seen.add(iid)
            picked.append(item)
            if len(picked) >= 50:
                break
        page += 1
        time.sleep(0.2)

    manifest = []
    for i, item in enumerate(picked, start=1):
        url = photo_url(item)
        ext = ".jpg"
        dest = OUT / f"{i:02d}.jpg"
        try:
            download(op, url, dest)
            ok = dest.stat().st_size > 1000
        except (urllib.error.URLError, TimeoutError) as e:
            ok = False
            dest.write_text("") if dest.exists() else None
            print(f"fail {i} {item.get('title')} {e}", flush=True)
            continue
        rec = {
            "n": i,
            "id": item["id"],
            "title": item.get("title"),
            "path": str(dest),
            "url": item.get("url"),
            "photo_url": url,
            "bytes": dest.stat().st_size if ok else 0,
        }
        manifest.append(rec)
        print(f"ok {i:02d} {item.get('title')}", flush=True)
        time.sleep(0.05)

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"DONE n={len(manifest)} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
