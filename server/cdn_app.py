#!/usr/bin/env python3
"""Nezopt cardscan identify: TCG YOLO + CDN-Milo 128-d.

Identity is CardTrader ct_id. Public page is ct_id * 2.
Does not write CLIP, cdn_images, or Flutter milo_index.
Keeps the GPU free (YOLO+Milo on CPU).
"""
from __future__ import annotations

import json
import os
import threading
import time
from io import BytesIO
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "-1")

# Defaults are relative to this checkout: server/ lives in BattleScan/, and the
# PokoinTest index is a sibling project (Startup/Pokoin/PokoinTest).
BATTLESCAN_ROOT = Path(__file__).resolve().parent.parent
POKOINTEST_ROOT = BATTLESCAN_ROOT.parent / "PokoinTest"

WEB = Path(
    os.environ.get(
        "CARDSCAN_WEB",
        str(BATTLESCAN_ROOT / "web"),
    )
)

INDEX = Path(
    os.environ.get(
        "CDN_MILO",
        str(POKOINTEST_ROOT / "index" / "cdn_milo"),
    )
)
MILO_PATH = Path(
    os.environ.get(
        "MILO_ONNX",
        str(BATTLESCAN_ROOT / "clones/CollectorVision/collector_vision/weights/milo.onnx"),
    )
)
YOLO_PT = Path(
    os.environ.get(
        "YOLO_PT",
        str(BATTLESCAN_ROOT / "clones/TCG/backend/models/card_detector.pt"),
    )
)

YOLO_CONF = 0.25
YOLO_IOU = 0.45
MIN_AREA = 0.01
MILO_SIZE = 448
MILO_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
MILO_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MAX_BYTES = 4 * 1024 * 1024
ORT_THREADS = int(os.environ.get("CARDSCAN_THREADS", "4"))

_lock = threading.Lock()
_yolo = None
_milo = None
_vecs: np.ndarray | None = None
_cards: list[dict] = []


def _load() -> None:
    global _yolo, _milo, _vecs, _cards
    from ultralytics import YOLO
    import onnxruntime as ort

    _yolo = YOLO(str(YOLO_PT))
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = ORT_THREADS
    opts.inter_op_num_threads = 1
    _milo = ort.InferenceSession(str(MILO_PATH), opts, providers=["CPUExecutionProvider"])
    raw = np.load(INDEX / "embeddings.npy")
    if raw.ndim != 2 or raw.shape[1] != 128:
        raise RuntimeError(f"cdn_milo embeddings shape {raw.shape}")
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    _vecs = (raw / np.clip(norms, 1e-8, None)).astype(np.float32)
    recs = []
    for line in (INDEX / "metadata.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        ct = str(o.get("ct_id") or "").strip()
        public = str(o.get("public_id") or "").strip()
        if not ct.isdigit():
            continue
        if not public.isdigit():
            public = str(int(ct) * 2)
        recs.append(
            {
                "id": ct,
                "ct_id": ct,
                "public_id": public,
                "name": o.get("name") or "",
                "collector_number": o.get("n") or None,
                "set": o.get("set") or None,
                "pokoin_url": f"https://pokoin.com/{public}",
            }
        )
    if len(recs) != _vecs.shape[0]:
        raise RuntimeError(f"metadata {len(recs)} vs embeddings {_vecs.shape[0]}")
    _cards = recs


def _detect(rgb: np.ndarray) -> list[dict]:
    h, w = rgb.shape[:2]
    bgr = rgb[:, :, ::-1].copy()
    result = _yolo.predict(
        bgr,
        verbose=False,
        conf=YOLO_CONF,
        iou=YOLO_IOU,
        device="cpu",
        imgsz=640,
    )[0]
    min_area = MIN_AREA * w * h
    raw = []
    for box in result.boxes:
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
        if (x2 - x1) * (y2 - y1) < min_area:
            continue
        raw.append(
            {
                "xyxy": [round(v, 1) for v in (x1, y1, x2, y2)],
                "conf": round(float(box.conf[0]), 4),
            }
        )
    raw.sort(key=lambda b: b["conf"], reverse=True)
    return raw


def _orientations(crop_rgb: np.ndarray) -> list[np.ndarray]:
    h, w = crop_rgb.shape[:2]
    if h < 8 or w < 8:
        return [crop_rgb]
    aspect = w / max(h, 1)

    def rot90_cw(arr: np.ndarray) -> np.ndarray:
        return np.rot90(arr, k=-1)

    r90 = rot90_cw(crop_rgb)
    r180 = rot90_cw(r90)
    r270 = rot90_cw(r180)
    if 0.85 <= aspect <= 1.18:
        return [crop_rgb, r90, r180, r270]
    if w > h:
        return [r90, r270]
    return [crop_rgb, r180]


def _embed(crop_rgb: np.ndarray) -> np.ndarray:
    im = Image.fromarray(crop_rgb).resize((MILO_SIZE, MILO_SIZE), Image.BILINEAR)
    x = np.asarray(im, dtype=np.float32) / 255.0
    x = (x - MILO_MEAN) / MILO_STD
    x = np.transpose(x, (2, 0, 1))[None, ...]
    name = _milo.get_inputs()[0].name
    vec = np.asarray(_milo.run(None, {name: x})[0], dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(vec))
    return vec / max(n, 1e-8)


def _search(q: np.ndarray, top_k: int) -> list[dict]:
    scores = _vecs @ q
    k = min(top_k, scores.shape[0])
    idx = np.argpartition(scores, -k)[-k:]
    idx = idx[np.argsort(scores[idx])[::-1]]
    hits = []
    for i in idx:
        rec = dict(_cards[int(i)])
        rec["score"] = round(float(scores[int(i)]), 4)
        hits.append(rec)
    return hits


def _embed_best(crop_rgb: np.ndarray, top_k: int, live: bool = False) -> tuple[list[dict], int]:
    best: list[dict] = []
    best_score = -1.0
    tried = 0
    orients = _orientations(crop_rgb)
    for oriented in orients:
        tried += 1
        hits = _search(_embed(oriented), top_k)
        score = float(hits[0]["score"]) if hits else -1.0
        if score > best_score:
            best_score = score
            best = hits
        if best_score >= 0.82:
            break
    return best, tried


def _box_fills_center(box: dict, img_w: int, img_h: int) -> bool:
    if img_w <= 0 or img_h <= 0:
        return False
    x1, y1, x2, y2 = (float(v) for v in box["xyxy"])
    width, height = x2 - x1, y2 - y1
    if width < 48 or height < 48:
        return False
    aspect = width / max(height, 1e-6)
    portrait = 0.58 <= aspect <= 0.88
    landscape = (1 / 0.88) <= aspect <= (1 / 0.58)
    if not portrait and not landscape:
        return False
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    if abs(cx - 0.5) > 0.18 or abs(cy - 0.5) > 0.18:
        return False
    return (width * height) / (img_w * img_h) >= 0.12


def _immediate(hits: list[dict], boxes: list[dict], img_w: int, img_h: int) -> bool:
    if not hits or not boxes:
        return False
    if float(hits[0].get("score") or 0) < 0.60:
        return False
    if boxes:
        return True
    return False


app = FastAPI(title="pokoin-cardscan-nez", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://pokoin.com",
        "https://www.pokoin.com",
        "https://cardscan.pokoin.com",
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    _load()


@app.api_route("/", methods=["GET", "HEAD"])
def index():
    page = WEB / "index.html"
    if not page.is_file():
        raise HTTPException(404, "scanner page missing")
    return FileResponse(page, media_type="text/html; charset=utf-8")


@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    return {
        "ok": True,
        "host": "nezopt",
        "n": 0 if _vecs is None else int(_vecs.shape[0]),
        "identity": "ct_id",
        "detect": "yolo-pt-cpu",
        "identify": "cdn-milo-128",
        "orient": "90-270",
        "index": "cdn_milo",
    }


@app.post("/identify")
async def identify(
    file: UploadFile = File(...),
    top_k: int = 5,
    live: bool = Query(False),
):
    if _vecs is None:
        raise HTTPException(503, "models not loaded")
    top_k = max(1, min(int(top_k), 3 if live else 8))
    blob = await file.read()
    if not blob or len(blob) > MAX_BYTES:
        raise HTTPException(400, f"image must be 1–{MAX_BYTES} bytes")
    try:
        im = Image.open(BytesIO(blob)).convert("RGB")
    except Exception as exc:
        raise HTTPException(400, "could not decode image") from exc
    w, h = im.size
    long_edge = max(w, h)
    if long_edge > 1600:
        scale = 1600 / long_edge
        im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    rgb = np.asarray(im)
    if live:
        got_lock = _lock.acquire(blocking=False)
        if not got_lock:
            return JSONResponse({"ok": True, "busy": True, "identity": "ct_id", "host": "nezopt"})
    else:
        _lock.acquire()
    t0 = time.perf_counter()
    try:
        boxes = _detect(rgb)
        detect_ms = (time.perf_counter() - t0) * 1000
        hits = []
        identify_ms = 0.0
        n_orient = 0
        if boxes:
            x1, y1, x2, y2 = (int(round(v)) for v in boxes[0]["xyxy"])
            crop = rgb[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
            if crop.size:
                t1 = time.perf_counter()
                hits, n_orient = _embed_best(crop, top_k, live=live)
                identify_ms = (time.perf_counter() - t1) * 1000
    finally:
        _lock.release()
    h, w = rgb.shape[:2]
    top = hits[0] if hits else None
    print(
        f"identify live={int(live)} detect={detect_ms:.0f} milo={identify_ms:.0f} "
        f"orient={n_orient} ct={(top or {}).get('ct_id', '-')} "
        f"score={(top or {}).get('score', 0)} pokoin={(top or {}).get('pokoin_url', '')}",
        flush=True,
    )
    return JSONResponse(
        {
            "ok": True,
            "host": "nezopt",
            "identity": "ct_id",
            "detect_ms": round(detect_ms, 1),
            "identify_ms": round(identify_ms, 1),
            "img_w": int(w),
            "img_h": int(h),
            "orientations": n_orient,
            "immediate": _immediate(hits, boxes, w, h),
            "boxes": boxes,
            "hits": hits,
            "top1": top,
        }
    )
