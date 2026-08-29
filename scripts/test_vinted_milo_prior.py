#!/usr/bin/env python3
"""Vinted 50: CLIP + Milo prior vs listing-title overlap (no OCR)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path("/home/nez/Projects/BattleScan")
SCRIPTS = Path("/home/nez/Projects/pokoin/PokoinTest/scripts")
INDEX = Path("/home/nez/Projects/pokoin/PokoinTest/index/blueprint_clip")
KEEP = ROOT / "results/vinted-keepers.json"
BOXES = ROOT / "results/yolo-boxes-vinted.json"
MANIFEST = ROOT / "images/vinted/manifest.json"


def crop(im: Image.Image, box: dict | None) -> Image.Image:
    if not box:
        return im
    x1, y1, x2, y2 = (int(round(v)) for v in box["xyxy"])
    w, h = im.size
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return im
    return im.crop((x1, y1, x2, y2))


def main() -> None:
    sys.path.insert(0, str(SCRIPTS))
    import torch
    import open_clip
    import match_blueprint_clip as clipm

    keepers = {r["photo"]: r for r in json.loads(KEEP.read_text())}
    boxes = json.loads(BOXES.read_text())
    titles = {Path(r["path"]).name: r.get("title") or "" for r in json.loads(MANIFEST.read_text())}
    vecs, meta = clipm.load_index(INDEX)
    number_index = clipm.build_number_index(meta)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fp16 = device == "cuda"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model = model.to(device).eval()
    if fp16:
        model = model.half()
    print(f"device={device} fp16={fp16} n={len(keepers)}", flush=True)

    rows = []
    t0 = time.perf_counter()
    for name, rec in keepers.items():
        path = ROOT / "images/vinted" / name
        im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        bl = (boxes.get(name) or {}).get("boxes") or []
        rgb = crop(im, bl[0] if bl else None)
        qn = clipm.encode(model, preprocess, device, rgb, fp16=fp16)
        cv = rec.get("cv_top1") or {}
        prior = clipm.prior_from_milo(cv.get("name"), cv.get("collector_number"), cv.get("set"))
        only = clipm.hits_from(vecs, meta, qn, 5, [])
        milo = clipm.hits_from(vecs, meta, qn, 5, [], ocr=prior, number_index=number_index)
        title = titles.get(name, "")
        row = {
            "photo": name,
            "title": title,
            "cv": cv.get("name"),
            "clip_only": only["hits"][0]["name"] if only["hits"] else None,
            "clip_milo": milo["hits"][0]["name"] if milo["hits"] else None,
            "overlap_only": round(clipm.title_overlap(title, only["hits"][0]["name"] if only["hits"] else ""), 3),
            "overlap_milo": round(clipm.title_overlap(title, milo["hits"][0]["name"] if milo["hits"] else ""), 3),
        }
        rows.append(row)
        print(
            json.dumps({k: row[k] for k in ("photo", "cv", "clip_milo", "overlap_milo")}, ensure_ascii=False),
            flush=True,
        )
    elapsed = time.perf_counter() - t0
    n = len(rows)
    summary = {
        "elapsed_s": round(elapsed, 2),
        "per_photo_ms": round(elapsed / n * 1000, 1) if n else None,
        "title_overlap_clip_only": f"{sum(1 for r in rows if r['overlap_only'] > 0)}/{n}",
        "title_overlap_clip_milo": f"{sum(1 for r in rows if r['overlap_milo'] > 0)}/{n}",
        "rows": rows,
    }
    dest = ROOT / "results/vinted-clip-milo-prior.json"
    dest.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps({k: summary[k] for k in summary if k != "rows"}), flush=True)


if __name__ == "__main__":
    main()
