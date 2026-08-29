#!/usr/bin/env python3
"""One-process cold timing so Ultralytics compile is not shared."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path("/home/nez/Projects/BattleScan")
PHOTO = ROOT / "images/01-pikachu-sm86-slab.png"
which = sys.argv[1]


def ms(s: float) -> float:
    return round(s * 1000.0, 1)


img = cv2.imread(str(PHOTO))
out = {}

if which == "tcg":
    from ultralytics import YOLO

    t0 = time.perf_counter()
    model = YOLO(str(ROOT / "clones/TCG/backend/models/card_detector.pt"))
    load = time.perf_counter() - t0
    t1 = time.perf_counter()
    model(img, verbose=False)
    infer = time.perf_counter() - t1
    out = {"scanner": "qtran TCG YOLO", "load_ms": ms(load), "cold_infer_ms": ms(infer), "cold_total_ms": ms(load + infer)}
elif which == "shrey":
    from ultralytics import YOLO

    t0 = time.perf_counter()
    model = YOLO(
        str(
            ROOT
            / "clones/Pokemon-Card-Scanning-Webapp/detector_models/pokemon_detector4/weights/best.pt"
        )
    )
    load = time.perf_counter() - t0
    t1 = time.perf_counter()
    model(img, verbose=False)
    infer = time.perf_counter() - t1
    out = {"scanner": "Shrey YOLO", "load_ms": ms(load), "cold_infer_ms": ms(infer), "cold_total_ms": ms(load + infer)}
elif which == "cv":
    sys.path.insert(0, str(ROOT / "clones/CollectorVision"))
    import collector_vision as cvg
    from PIL import Image

    t0 = time.perf_counter()
    catalog = cvg.Catalog.load("pokemon")
    load = time.perf_counter() - t0
    x1, y1, x2, y2 = 198, 251, 411, 523
    crop = img[y1:y2, x1:x2]
    pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    t1 = time.perf_counter()
    catalog.search_records(catalog.embedder.embed(pil), top_k=8)
    infer = time.perf_counter() - t1
    out = {
        "scanner": "CollectorVision",
        "catalog_load_ms": ms(load),
        "cold_embed_search_ms": ms(infer),
        "cold_total_ms": ms(load + infer),
    }
else:
    raise SystemExit("usage: cold_isolate.py tcg|shrey|cv")

print(json.dumps(out))
