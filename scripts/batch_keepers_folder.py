#!/usr/bin/env python3
"""One-load TCG YOLO + Shrey YOLO + CollectorVision on a photo folder."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path("/home/nez/Projects/BattleScan")
TCG_WEIGHTS = ROOT / "clones/TCG/backend/models/card_detector.pt"
SHREY_WEIGHTS = (
    ROOT
    / "clones/Pokemon-Card-Scanning-Webapp/detector_models/pokemon_detector4/weights/best.pt"
)


def boxes_from(result) -> list[dict]:
    out = []
    for b in result.boxes:
        out.append(
            {
                "xyxy": [round(x, 2) for x in b.xyxy[0].cpu().numpy().tolist()],
                "conf": round(float(b.conf[0]), 4),
                "cls": int(b.cls[0]),
            }
        )
    out.sort(key=lambda x: x["conf"], reverse=True)
    return out


def crop_box(img: np.ndarray, box: dict) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in box["xyxy"])
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return img[y1:y2, x1:x2]


def hit_dict(hit) -> dict:
    if hasattr(hit, "__dict__"):
        raw = dict(hit.__dict__)
    elif isinstance(hit, dict):
        raw = dict(hit)
    else:
        raw = {"repr": str(hit)}
    meta = raw.get("metadata") or {}
    return {
        "name": raw.get("name"),
        "score": float(raw.get("score") or 0),
        "collector_number": meta.get("collector_number"),
        "set": meta.get("set"),
        "key": raw.get("key") or raw.get("id"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", type=Path, required=True)
    ap.add_argument("--glob", default="*.jpg")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--manifest", type=Path)
    args = ap.parse_args()

    from ultralytics import YOLO

    sys.path.insert(0, str(ROOT / "clones/CollectorVision"))
    import collector_vision as cvg

    files = sorted(args.photos.glob(args.glob))
    titles = {}
    if args.manifest:
        for rec in json.loads(args.manifest.read_text()):
            titles[Path(rec["path"]).name] = rec.get("title") or ""

    print("load YOLO + CollectorVision catalog…", flush=True)
    t0 = time.perf_counter()
    tcg_m = YOLO(str(TCG_WEIGHTS))
    shrey_m = YOLO(str(SHREY_WEIGHTS))
    catalog = cvg.Catalog.load("pokemon")
    print(f"loaded in {time.perf_counter() - t0:.1f}s n={len(files)}", flush=True)

    rows = []
    for path in files:
        img = cv2.imread(str(path))
        if img is None:
            rows.append({"photo": path.name, "error": "unreadable"})
            continue
        tcg_boxes = boxes_from(tcg_m(img, verbose=False)[0])
        shrey_boxes = boxes_from(shrey_m(img, verbose=False)[0])
        box = tcg_boxes[0] if tcg_boxes else None
        cv_hits = []
        if box is not None:
            crop = crop_box(img, box)
            pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            emb = catalog.embedder.embed(pil)
            cv_hits = [hit_dict(h) for h in catalog.search_records(emb, top_k=5)]
        rec = {
            "photo": path.name,
            "title": titles.get(path.name),
            "tcg": {"n_boxes": len(tcg_boxes), "top_conf": tcg_boxes[0]["conf"] if tcg_boxes else None},
            "shrey": {"n_boxes": len(shrey_boxes), "top_conf": shrey_boxes[0]["conf"] if shrey_boxes else None},
            "cv_top1": cv_hits[0] if cv_hits else None,
            "cv_hits": cv_hits,
        }
        rows.append(rec)
        top = rec["cv_top1"]
        print(
            f"{path.name} tcg={rec['tcg']['n_boxes']} shrey={rec['shrey']['n_boxes']} "
            f"cv={None if not top else top.get('name')}",
            flush=True,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"DONE n={len(rows)} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
