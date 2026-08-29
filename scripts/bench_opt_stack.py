#!/usr/bin/env python3
"""Measure stack knobs that actually change Fast/Multi latency.

Steps measured after warmup, median of 5:
  1. TCG YOLO .pt vs ONNX (imgsz=640)
  2. Milo ONNX: graph opt + intra_op thread sweep
  3. RapidOCR: 5 crops (old Multi) vs 2 crops (name+collector)
  4. Search: NumPy exact cosine (proves FAISS is not the bottleneck)

Usage:
  BattleScan/.venv/bin/python scripts/bench_opt_stack.py
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
TCG_PT = ROOT / "clones/TCG/backend/models/card_detector.pt"
ONNX_OUT = ROOT / "models/card_detector.onnx"
MILO = ROOT / "clones/CollectorVision/collector_vision/weights/milo.onnx"
PHOTO = PHOTOS / "03-dark-omanyte-neo-destiny.png"
WARMUP = 1
REPEATS = 5


def median_ms(samples: list[float]) -> float:
    return round(statistics.median(samples) * 1000.0, 2)


def time_fn(fn, n: int = REPEATS) -> float:
    for _ in range(WARMUP):
        fn()
    xs = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        xs.append(time.perf_counter() - t)
    return median_ms(xs)


def crop_xyxy(img, xyxy):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return img[y1:y2, x1:x2]


def ensure_yolo_onnx() -> Path | None:
    if ONNX_OUT.exists() and ONNX_OUT.stat().st_size > 1000:
        return ONNX_OUT
    ONNX_OUT.parent.mkdir(parents=True, exist_ok=True)
    from ultralytics import YOLO

    print("export YOLO ONNX…", flush=True)
    model = YOLO(str(TCG_PT))
    exported = model.export(format="onnx", imgsz=640, simplify=True, nms=True)
    src = Path(str(exported))
    if src.resolve() != ONNX_OUT.resolve():
        ONNX_OUT.write_bytes(src.read_bytes())
        data = src.with_suffix(".onnx.data")
        if data.exists():
            dest_data = ONNX_OUT.with_suffix(".onnx.data")
            dest_data.write_bytes(data.read_bytes())
    return ONNX_OUT if ONNX_OUT.exists() else src


def bench_yolo(img_bgr) -> dict:
    from ultralytics import YOLO

    rows = {}
    for label, path in (("pt", TCG_PT), ("onnx", ensure_yolo_onnx())):
        if path is None or not Path(path).exists():
            continue
        model = YOLO(str(path))
        rows[label] = {
            "median_ms": time_fn(lambda m=model: model(img_bgr, verbose=False, conf=0.25, iou=0.45)),
            "path": str(path),
        }
        print(f"yolo_{label}={rows[label]['median_ms']}ms", flush=True)
    return rows


def bench_milo(pil: Image.Image) -> dict:
    import onnxruntime as ort
    from collector_vision.embedders.neural import _preprocess_pil

    x = _preprocess_pil(pil, 448)
    rows = []
    for opt_name, opt in (
        ("default", None),
        ("all", ort.GraphOptimizationLevel.ORT_ENABLE_ALL),
        ("extended", ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED),
    ):
        for threads in (1, 2, 4, 8, 12):
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = threads
            opts.inter_op_num_threads = 1
            if opt is not None:
                opts.graph_optimization_level = opt
            sess = ort.InferenceSession(
                str(MILO),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            name = sess.get_inputs()[0].name

            def run(s=sess, n=name):
                s.run(None, {n: x})

            rec = {
                "opt": opt_name,
                "threads": threads,
                "embed_median_ms": time_fn(run),
            }
            rows.append(rec)
            print(json.dumps(rec), flush=True)
    best = min(rows, key=lambda r: r["embed_median_ms"])
    return {"best": best, "grid": rows}


def bench_search() -> dict:
    sys.path.insert(0, str(ROOT / "clones/CollectorVision"))
    import collector_vision as cvg

    catalog = cvg.Catalog.load("pokemon")
    vecs = catalog.embeddings.astype(np.float32)
    nrm = np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-8, None)
    vecs = vecs / nrm
    q = vecs[0:1]

    def search():
        sims = vecs @ q.T
        return int(np.argmax(sims))

    return {
        "n": int(vecs.shape[0]),
        "dim": int(vecs.shape[1]),
        "exact_cosine_median_ms": time_fn(search, n=20),
    }


def bench_ocr(crop_bgr) -> dict:
    from rapidocr_onnxruntime import RapidOCR

    engine = RapidOCR()
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    name = rgb[int(h * 0.05) : int(h * 0.22), int(w * 0.05) : int(w * 0.95)]
    collector = rgb[int(h * 0.78) : h, int(w * 0.05) : int(w * 0.55)]
    crops5 = [name, collector, name, collector, rgb]
    crops2 = [name, collector]

    def run(crops):
        for c in crops:
            engine(c)

    return {
        "five_crop_median_ms": time_fn(lambda: run(crops5), n=3),
        "two_crop_median_ms": time_fn(lambda: run(crops2), n=3),
    }


def main() -> None:
    sys.path.insert(0, str(ROOT / "clones/CollectorVision"))
    img = cv2.imread(str(PHOTO))
    from ultralytics import YOLO

    yolo = YOLO(str(TCG_PT))
    box = yolo(img, verbose=False)[0].boxes
    xyxy = box[0].xyxy[0].cpu().numpy().tolist()
    crop = crop_xyxy(img, xyxy)
    pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

    out = {
        "photo": PHOTO.name,
        "cpu_count": __import__("os").cpu_count(),
        "yolo": bench_yolo(img),
        "milo": bench_milo(pil),
        "search": bench_search(),
        "ocr": bench_ocr(crop),
    }
    dest = ROOT / "results/bench-opt-stack.json"
    dest.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("yolo", "search", "ocr")}, indent=2))
    print(f"milo_best={out['milo']['best']} -> {dest}", flush=True)


if __name__ == "__main__":
    main()
