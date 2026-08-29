#!/usr/bin/env python3
"""Timed bake-off: TCG YOLO, Shrey YOLO, CollectorVision on a TCG crop.

Usage:
  .venv/bin/python scripts/run_keepers.py images/01-pikachu-sm86-slab.png
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/home/nez/Projects/BattleScan")
TCG_WEIGHTS = ROOT / "clones/TCG/backend/models/card_detector.pt"
SHREY_WEIGHTS = (
    ROOT
    / "clones/Pokemon-Card-Scanning-Webapp/detector_models/pokemon_detector4/weights/best.pt"
)
WARMUP = 1
REPEATS = 5


def ms(seconds: float) -> float:
    return round(seconds * 1000.0, 1)


def summarize(samples: list[float]) -> dict:
    return {
        "n": len(samples),
        "min_ms": ms(min(samples)),
        "median_ms": ms(statistics.median(samples)),
        "mean_ms": ms(statistics.fmean(samples)),
        "max_ms": ms(max(samples)),
        "samples_ms": [ms(s) for s in samples],
    }


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


def best_box(boxes: list[dict]) -> dict | None:
    return boxes[0] if boxes else None


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


def time_yolo(name: str, weights: Path, img: np.ndarray, out_dir: Path) -> dict:
    from ultralytics import YOLO

    t0 = time.perf_counter()
    model = YOLO(str(weights))
    load_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    first = model(img, verbose=False)[0]
    first_s = time.perf_counter() - t1
    boxes = boxes_from(first)

    for _ in range(WARMUP):
        model(img, verbose=False)

    samples = []
    last = first
    for _ in range(REPEATS):
        t = time.perf_counter()
        last = model(img, verbose=False)[0]
        samples.append(time.perf_counter() - t)

    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "detect.png"), last.plot())
    payload = {
        "scanner": name,
        "job": "detect",
        "weights": str(weights),
        "n_boxes": len(boxes),
        "boxes": boxes,
        "timing": {
            "load_ms": ms(load_s),
            "cold_infer_ms": ms(first_s),
            "cold_total_ms": ms(load_s + first_s),
            "warm_infer": summarize(samples),
        },
    }
    (out_dir / "result.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def time_collectorvision(img: np.ndarray, box: dict, out_dir: Path) -> dict:
    from PIL import Image

    sys.path.insert(0, str(ROOT / "clones/CollectorVision"))
    import collector_vision as cvg

    t0 = time.perf_counter()
    catalog = cvg.Catalog.load("pokemon")
    load_s = time.perf_counter() - t0

    crop = crop_box(img, box)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "crop.png"), crop)
    pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

    t1 = time.perf_counter()
    emb = catalog.embedder.embed(pil)
    hits_raw = catalog.search_records(emb, top_k=8)
    first_s = time.perf_counter() - t1
    hits = [hit_dict(h) for h in hits_raw]

    for _ in range(WARMUP):
        catalog.search_records(catalog.embedder.embed(pil), top_k=8)

    samples = []
    for _ in range(REPEATS):
        t = time.perf_counter()
        catalog.search_records(catalog.embedder.embed(pil), top_k=8)
        samples.append(time.perf_counter() - t)

    payload = {
        "scanner": "CollectorVision",
        "job": "identify",
        "mode": "precrop-tcg-box",
        "crop_box": box,
        "hits": hits,
        "top1": hits[0] if hits else None,
        "timing": {
            "catalog_load_ms": ms(load_s),
            "cold_embed_search_ms": ms(first_s),
            "cold_total_ms": ms(load_s + first_s),
            "warm_embed_search": summarize(samples),
        },
    }
    (out_dir / "result.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def combo_scan(
    tcg_warm_ms: float, cv_warm_ms: float, tcg: dict, cv: dict
) -> dict:
    return {
        "scanner": "TCG YOLO + CollectorVision",
        "job": "detect+identify",
        "timing": {
            "warm_scan_ms": round(tcg_warm_ms + cv_warm_ms, 1),
            "note": "median TCG infer + median CollectorVision embed/search; models already loaded",
        },
        "detect": {
            "n_boxes": tcg["n_boxes"],
            "top_conf": tcg["boxes"][0]["conf"] if tcg["boxes"] else None,
        },
        "identify": cv.get("top1"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("photo", type=Path)
    args = parser.parse_args()
    photo = args.photo.resolve()
    if not photo.is_file():
        print(f"missing photo: {photo}", file=sys.stderr)
        return 1

    img = cv2.imread(str(photo))
    if img is None:
        print(f"unreadable image: {photo}", file=sys.stderr)
        return 1
    h, w = img.shape[:2]
    stem = photo.stem
    res = ROOT / "results" / stem
    res.mkdir(parents=True, exist_ok=True)

    wall0 = time.perf_counter()
    tcg = time_yolo("qtran TCG YOLO", TCG_WEIGHTS, img, res / "TCG")
    shrey = time_yolo("Shrey YOLO", SHREY_WEIGHTS, img, res / "Shrey")
    box = best_box(tcg["boxes"])
    if box is None:
        cv = {"error": "TCG YOLO found no box", "hits": [], "top1": None, "timing": {}}
        (res / "CollectorVision").mkdir(parents=True, exist_ok=True)
        (res / "CollectorVision" / "result.json").write_text(
            json.dumps(cv, indent=2) + "\n"
        )
        combo = None
    else:
        cv = time_collectorvision(img, box, res / "CollectorVision")
        combo = combo_scan(
            tcg["timing"]["warm_infer"]["median_ms"],
            cv["timing"]["warm_embed_search"]["median_ms"],
            tcg,
            cv,
        )
    wall_s = time.perf_counter() - wall0

    summary = {
        "photo": str(photo),
        "size_px": [w, h],
        "bench_wall_ms": ms(wall_s),
        "note": (
            "Cold = model load + first inference. Warm = median of "
            f"{REPEATS} runs after {WARMUP} warmup. CPU."
        ),
        "keepers": [tcg, shrey, cv],
        "combo": combo,
    }
    (res / "timing.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
