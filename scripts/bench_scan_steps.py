#!/usr/bin/env python3
"""Step-by-step latency + accuracy on BattleScan photos.

Contract:
  detect = YOLO box (not Canny, not CV corners)
  fast identify = Milo 128-d ONNX + exact cosine brute force
  multi identify = CLIP 512-d (separate GPU process)
  never mix 268-d HOG, 128-d Milo, 512-d CLIP

Usage (BattleScan venv):
  .venv/bin/python scripts/bench_scan_steps.py
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path("/home/nez/Projects/BattleScan")
PHOTOS = ROOT / "images"
TCG_WEIGHTS = ROOT / "clones/TCG/backend/models/card_detector.pt"
WARMUP = 1
REPEATS = 5
GT_NAMES = {
    "01-pikachu-sm86-slab.png": ["pikachu"],
    "02-mew-expedition-it-toploader.png": ["mew"],
    "03-dark-omanyte-neo-destiny.png": ["omanyte"],
    "04-deusolourdo-paldea-ir-fr.png": ["dudunsparce"],
    "05-slowpoke-delta-species-it.png": ["slowpoke"],
    "06-goldeen-pbl-087.png": ["goldeen"],
}


def ms(s: float) -> float:
    return round(s * 1000.0, 2)


def median(xs: list[float]) -> float:
    return ms(statistics.median(xs))


def crop_box(img: np.ndarray, xyxy: list[float]) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return img[y1:y2, x1:x2]


def name_hit(top: str, expect: list[str]) -> bool:
    low = (top or "").lower()
    return any(e in low for e in expect)


def time_fn(fn, n: int) -> list[float]:
    for _ in range(WARMUP):
        fn()
    samples = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t)
    return samples


def main() -> None:
    sys.path.insert(0, str(ROOT / "clones/CollectorVision"))
    sys.path.insert(0, "/home/nez/Projects/pokoin/PokoinTest")
    from ultralytics import YOLO
    import collector_vision as cvg
    from pipeline import detect_card_quad

    files = [PHOTOS / n for n in GT_NAMES]
    imgs = {p.name: cv2.imread(str(p)) for p in files}

    print("load YOLO + Milo catalog…", flush=True)
    t0 = time.perf_counter()
    yolo = YOLO(str(TCG_WEIGHTS))
    catalog = cvg.Catalog.load("pokemon")
    _ = catalog.embedder
    load_s = time.perf_counter() - t0
    print(f"load_ms={ms(load_s)} catalog_n={len(catalog)} dim={catalog.embeddings.shape}", flush=True)

    rows = []
    pokemon_ok = 0
    for path in files:
        img = imgs[path.name]
        expect = GT_NAMES[path.name]

        def canny():
            detect_card_quad(img)

        canny_s = time_fn(canny, REPEATS)

        def detect():
            return yolo(img, verbose=False)[0]

        detect_s = time_fn(detect, REPEATS)
        last = yolo(img, verbose=False)[0]
        boxes = []
        for b in last.boxes:
            boxes.append(
                {
                    "xyxy": [round(x, 2) for x in b.xyxy[0].cpu().numpy().tolist()],
                    "conf": round(float(b.conf[0]), 4),
                }
            )
        boxes.sort(key=lambda x: x["conf"], reverse=True)
        box = boxes[0] if boxes else None
        crop = crop_box(img, box["xyxy"]) if box else img
        pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

        def embed():
            return catalog.embedder.embed(pil)

        embed_s = time_fn(embed, REPEATS)
        emb = catalog.embedder.embed(pil)

        def search():
            return catalog.search_records(emb, top_k=5)

        search_s = time_fn(search, REPEATS)
        hits_raw = catalog.search_records(emb, top_k=5)
        top_name = ""
        if hits_raw:
            h = hits_raw[0]
            raw = h.__dict__ if hasattr(h, "__dict__") else dict(h)
            top_name = str(raw.get("name") or "")
        ok = name_hit(top_name, expect)
        pokemon_ok += int(ok)
        rec = {
            "photo": path.name,
            "yolo_n": len(boxes),
            "yolo_conf": box["conf"] if box else None,
            "canny_median_ms": median(canny_s),
            "yolo_median_ms": median(detect_s),
            "milo_embed_median_ms": median(embed_s),
            "milo_search_median_ms": median(search_s),
            "fast_total_median_ms": round(
                median(detect_s) + median(embed_s) + median(search_s), 2
            ),
            "top1": top_name,
            "right_pokemon": ok,
        }
        rows.append(rec)
        print(json.dumps(rec, ensure_ascii=False), flush=True)

    out = {
        "note": (
            "Warm median of 5 after 1 warmup. Detect=TCG YOLO. Identify=Milo 128-d "
            "ONNX + exact cosine (no OCR, no FAISS). Search is O(N) matmul; N~108k is still ms."
        ),
        "load_ms": ms(load_s),
        "right_pokemon": f"{pokemon_ok}/{len(files)}",
        "steps": rows,
    }
    dest = ROOT / "results/bench-scan-steps.json"
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"right_pokemon={pokemon_ok}/{len(files)} -> {dest}", flush=True)


if __name__ == "__main__":
    main()
