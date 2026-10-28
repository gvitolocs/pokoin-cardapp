#!/usr/bin/env python3
"""Pokoin Fast identify API: TCG YOLO TFLite + Milo 128-d ONNX.

Legacy identity is TCGplayer; selected catalogs return public Pokoin IDs. Sized for Oracle
Always Free (1 GB). One inference at a time.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from io import BytesIO
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from fastapi.concurrency import run_in_threadpool
from catalogs import CatalogStore
import card_quad

ROOT = Path(os.environ.get("CARDSCAN_ROOT", Path(__file__).resolve().parent))
WEB = ROOT / "web"
MODELS = Path(os.environ.get("CARDSCAN_MODELS", ROOT / "models"))
YOLO_PATH = MODELS / "card_detector.tflite"
MILO_PATH = MODELS / "milo.onnx"
EMB_PATH = MODELS / "embeddings.bin"
META_PATH = MODELS / "metadata.jsonl"
CATALOG_ROOT = Path(os.environ.get("CARDSCAN_CATALOGS", ROOT / "catalogs"))
_catalogs = None

YOLO_SIZE = 640
YOLO_ANCHORS = 8400
YOLO_CONF = 0.25
YOLO_IOU = 0.45
MIN_AREA = 0.01
MILO_SIZE = 448
MILO_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
MILO_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MAX_BYTES = 4 * 1024 * 1024
ORT_THREADS = int(os.environ.get("CARDSCAN_THREADS", "1"))
# Optional ROCm onnxruntime tree (7900 XTX). Same layout as /tmp/pokoin-ort-rocm /
# western-gpu-ocr.py — PYTHONPATH is set by battlescan-fast.service when present.
ORT_PATH = Path(os.environ.get("CARDSCAN_ORT_PATH", "")).expanduser()
# Extension album lots: leftover-JPEG singles like marketplace /scan.
# Camera /identify without catalog stays TCGPlayer; clients must set catalog.
ALBUM_CATALOG = "pokemon_generic"
ALBUM_BOX_LIMIT = 24
ALBUM_PHOTO_LIMIT = 24
ALBUM_UNIQUE_MIN_SCORE = 0.50
ALBUM_EXTRA_MIN_SCORE = 0.80
ALBUM_DENSE_BOX_COUNT = 6
ALBUM_LOCK_TIMEOUT_S = 90.0

_lock = threading.Lock()
_yolo = None
_milo = None
_vecs: np.ndarray | None = None
_cards: list[dict] = []
_tcg_to_ct: dict[str, str] = {}
_milo_providers: list[str] = []


def _load_tflite(path: Path):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        from tflite_runtime.interpreter import Interpreter  # type: ignore
    it = Interpreter(model_path=str(path), num_threads=ORT_THREADS)
    it.allocate_tensors()
    return it


def _prepare_ort_path() -> None:
    """Prefer CARDSCAN_ORT_PATH (ROCm wheel) over the venv CPU onnxruntime."""
    if not ORT_PATH.is_dir():
        return
    root = str(ORT_PATH.resolve())
    # Must precede any `import onnxruntime` — the first import wins.
    while root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)


def _ort_providers() -> list:
    """Prefer ROCMExecutionProvider on nezopt 7900 XTX; fall back to CPU.

    Install path: CARDSCAN_ORT_PATH (BattleScan/runtime/ort-rocm). MIGraphX is
    skipped — Milo is fixed 448² but EP selection matches western-gpu-ocr.
    """
    _prepare_ort_path()
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    providers: list = []
    if "ROCMExecutionProvider" in available and os.environ.get("CARDSCAN_FORCE_CPU") != "1":
        providers.append(("ROCMExecutionProvider", {"device_id": int(os.environ.get("CARDSCAN_GPU", "0"))}))
    providers.append("CPUExecutionProvider")
    return providers


def _load() -> None:
    global _yolo, _milo, _vecs, _cards, _tcg_to_ct, _catalogs, _milo_providers
    _prepare_ort_path()
    import onnxruntime as ort

    _yolo = _load_tflite(YOLO_PATH)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = ORT_THREADS
    opts.inter_op_num_threads = 1
    providers = _ort_providers()
    _milo = ort.InferenceSession(str(MILO_PATH), opts, providers=providers)
    _milo_providers = list(_milo.get_providers())
    raw = np.fromfile(EMB_PATH, dtype=np.float32)
    if raw.size % 128:
        raise RuntimeError("embeddings.bin not divisible by 128")
    _vecs = raw.reshape(-1, 128)
    lines = META_PATH.read_text(encoding="utf-8").splitlines()
    recs = []
    for line in lines:
        if not line.strip():
            continue
        o = json.loads(line)
        recs.append(
            {
                "id": str(o.get("id") or ""),
                "name": o.get("name") or "",
                "collector_number": o.get("n") or None,
                "set": o.get("set") or None,
            }
        )
    if len(recs) != _vecs.shape[0]:
        raise RuntimeError(f"metadata {len(recs)} vs embeddings {_vecs.shape[0]}")
    _cards = recs
    tcg_path = MODELS / "pokoin_tcg_ids.json"
    mapped: dict[str, str] = {}
    if tcg_path.is_file():
        raw_map = json.loads(tcg_path.read_text(encoding="utf-8"))
        if isinstance(raw_map, dict):
            mapped = {str(k): str(v) for k, v in raw_map.items() if str(v).isdigit()}
    _tcg_to_ct = mapped
    _catalogs = CatalogStore(CATALOG_ROOT, MILO_PATH)


def _nms(boxes: list[tuple[float, float, float, float, float]]) -> list:
    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
    kept = []
    for b in boxes:
        if all(_iou(b, k) <= YOLO_IOU for k in kept):
            kept.append(b)
    return kept


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ba = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + ba - inter)


def _letterbox(rgb: np.ndarray, size: int = YOLO_SIZE) -> tuple[np.ndarray, float, float, float]:
    """Fit rgb into size×size with gray pad (same as Flutter letterboxYolo)."""
    h, w = rgb.shape[:2]
    scale = min(size / max(w, 1), size / max(h, 1))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = np.asarray(
        Image.fromarray(rgb).resize((nw, nh), Image.BILINEAR),
        dtype=np.uint8,
    )
    pad_x = (size - nw) // 2
    pad_y = (size - nh) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[pad_y : pad_y + nh, pad_x : pad_x + nw] = resized
    return canvas, scale, float(pad_x), float(pad_y)


def _detect(rgb: np.ndarray) -> list[dict]:
    h, w = rgb.shape[:2]
    square, scale, pad_x, pad_y = _letterbox(rgb, YOLO_SIZE)
    inp = np.asarray(square, dtype=np.float32) / 255.0
    inp = np.expand_dims(inp, 0)
    det = _yolo.get_input_details()[0]
    outd = _yolo.get_output_details()[0]
    _yolo.set_tensor(det["index"], inp)
    _yolo.invoke()
    out = _yolo.get_tensor(outd["index"])[0]  # (5, 8400)
    inv = 1.0 / max(scale, 1e-8)
    min_area = MIN_AREA * w * h
    raw = []
    for j in range(YOLO_ANCHORS):
        conf = float(out[4, j])
        if conf < YOLO_CONF:
            continue
        cx, cy, bw, bh = (float(out[k, j]) for k in range(4))
        x1 = max(0.0, min(w, (cx - bw / 2 - pad_x) * inv))
        y1 = max(0.0, min(h, (cy - bh / 2 - pad_y) * inv))
        x2 = max(0.0, min(w, (cx + bw / 2 - pad_x) * inv))
        y2 = max(0.0, min(h, (cy + bh / 2 - pad_y) * inv))
        if (x2 - x1) * (y2 - y1) < min_area:
            continue
        raw.append((x1, y1, x2, y2, conf))
    kept = _nms(raw)
    return [
        {
            "xyxy": [round(v, 1) for v in b[:4]],
            "conf": round(b[4], 4),
        }
        for b in kept
    ]


def _orientations(crop_rgb: np.ndarray) -> list[np.ndarray]:
    """Match native Milo: landscape 90/270, portrait 0/180, near-square all four."""
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


def _embed_best(
    crop_rgb: np.ndarray,
    top_k: int,
    early: float = 0.82,
    live: bool = False,
    catalog: str = "tcgplayer",
    timings: dict | None = None,
) -> tuple[list[dict], int]:
    best: list[dict] = []
    best_score = -1.0
    tried = 0
    orients = _orientations(crop_rgb)
    if live:
        orients = orients[:1]
    for oriented in orients:
        tried += 1
        started=time.perf_counter()
        vector=_embed(oriented)
        embedded=time.perf_counter()
        hits = _search(vector, top_k, catalog)
        if timings is not None:
            timings["embed_ms"] += (embedded-started)*1000
            timings["search_ms"] += (time.perf_counter()-embedded)*1000
        score = float(hits[0]["score"]) if hits else -1.0
        if score > best_score:
            best_score = score
            best = hits
        if best_score >= early:
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


def _species_token(name: str) -> str:
    cleaned = (
        str(name or "")
        .lower()
        .replace("(", " ")
        .replace(")", " ")
    )
    if " - " in cleaned:
        cleaned = cleaned.split(" - ", 1)[0]
    parts = [p for p in "".join(ch if ch.isalnum() else " " for ch in cleaned).split() if p]
    skip = {"ex", "gx", "v", "vmax", "vstar", "lvx", "mega"}
    for part in parts:
        if part not in skip:
            return part
    return parts[0] if parts else ""


def _immediate(hits: list[dict], boxes: list[dict], img_w: int, img_h: int) -> bool:
    if not hits or not boxes:
        return False
    top = hits[0]
    if float(top.get("score") or 0) < 0.80:
        return False
    if len(hits) >= 2 and float(top["score"]) - float(hits[1]["score"]) < 0.08:
        a, b = _species_token(top.get("name")), _species_token(hits[1].get("name"))
        if a and b and a != b:
            return False
    return _box_fills_center(boxes[0], img_w, img_h)


def _embed(crop_rgb: np.ndarray) -> np.ndarray:
    im = Image.fromarray(crop_rgb).resize((MILO_SIZE, MILO_SIZE), Image.BILINEAR)
    x = np.asarray(im, dtype=np.float32) / 255.0
    x = (x - MILO_MEAN) / MILO_STD
    x = np.transpose(x, (2, 0, 1))[None, ...]
    name = _milo.get_inputs()[0].name
    vec = np.asarray(_milo.run(None, {name: x})[0], dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(vec))
    return vec / max(n, 1e-8)


def _is_card_back_hit(rec: dict | None) -> bool:
    if not rec:
        return False
    if str(rec.get("item_kind") or "") == "card_back":
        return True
    compact = "".join(ch for ch in str(rec.get("name") or "").lower() if ch.isalnum())
    return "pokemoncardback" in compact or compact == "cardback"


def _best_card_back_score(scores: np.ndarray, cards: list[dict]) -> tuple[float, int]:
    best_i = -1
    best_score = -1.0
    for i, card in enumerate(cards):
        if not _is_card_back_hit(card):
            continue
        score = float(scores[i])
        if score > best_score:
            best_score = score
            best_i = i
    return best_score, best_i


def _prefer_card_back_hits(
    q: np.ndarray,
    hits: list[dict],
    catalog: str,
    scores: np.ndarray,
    cards: list[dict],
) -> list[dict]:
    """Marketplace card backs must beat Horsea/Froakie lookalike fronts.

    Compares the query against every card_back catalog row (small set), then
    promotes the best back when its score is strong or close to the current top1.
    """
    if catalog == "tcgplayer" or not len(cards):
        return hits
    best_score, best_i = _best_card_back_score(scores, cards)
    if best_i < 0:
        return hits
    for hit in hits:
        hit["_card_back_score"] = round(float(best_score), 4)
    if best_score < 0.45:
        return hits
    best = dict(cards[best_i])
    best["score"] = round(best_score, 4)
    best["_card_back_score"] = round(best_score, 4)
    top_score = float(hits[0].get("score") or 0) if hits else 0.0
    if hits and _is_card_back_hit(hits[0]) and top_score + 1e-6 >= best_score:
        return hits
    # Promote only when the back is strong, or nearly ties the raw top1 lookalike.
    if best_score >= 0.58 or (hits and best_score >= 0.48 and (top_score - best_score) <= 0.10):
        rest = [hit for hit in hits if not _is_card_back_hit(hit)]
        return [best] + rest
    return hits


def _search(q: np.ndarray, top_k: int, catalog: str = "tcgplayer") -> list[dict]:
    vectors, cards = (_vecs, _cards) if catalog == "tcgplayer" else _catalogs.get(catalog)
    scores = vectors @ q
    k = min(top_k, scores.shape[0])
    idx = np.argpartition(scores, -k)[-k:]
    idx = idx[np.argsort(scores[idx])[::-1]]
    hits = []
    for i in idx:
        rec = dict(cards[int(i)])
        rec["score"] = round(float(scores[int(i)]), 4)
        hits.append(_with_pokoin(rec) if catalog == "tcgplayer" else rec)
    if catalog != "tcgplayer":
        hits = _prefer_card_back_hits(q, hits, catalog, scores, cards)
    return hits


def _should_skip_card_back_box(hits: list[dict], catalog: str) -> bool:
    """Skip boxes that are western TCG backs even when Horsea/Froakie won top1."""
    if catalog == "tcgplayer" or not hits:
        return False
    top = hits[0]
    back_score = float(top.get("_card_back_score") or 0)
    top_score = float(top.get("score") or 0)
    if _is_card_back_hit(top) and top_score >= 0.48:
        return True
    # Strong absolute back match.
    if back_score >= 0.58:
        return True
    # Lookalike front barely beats the back gallery (typical Vinted back photos).
    if back_score >= 0.50 and top_score > 0 and (top_score - back_score) <= 0.08:
        return True
    # Weak raw top1 (<0.62) with a competitive back score — never a real front.
    # Real Horsea fronts score ~1.0 with card_back_score ~0.46, so they pass.
    if top_score < 0.62 and back_score >= 0.45 and (top_score - back_score) <= 0.14:
        return True
    return False


def _with_pokoin(rec: dict) -> dict:
    # Fast id is TCGplayer. Public page is ct_id × 2. Never double the TCG id.
    ct = _tcg_to_ct.get(str(rec.get("id") or ""))
    if not ct:
        return rec
    rec["ct_id"] = ct
    rec["public_id"] = str(int(ct) * 2)
    rec["pokoin_url"] = f"https://pokoin.com/{rec['public_id']}"
    return rec


app = FastAPI(title="pokoin-cardscan", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://pokoin.com",
        "https://www.pokoin.com",
        "https://cardscan.pokoin.com",
        "https://scan.pokoin.com",
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    _load()
    # Pay first-run graph setup before advertising a healthy inference worker.
    _detect(np.zeros((YOLO_SIZE,YOLO_SIZE,3),dtype=np.uint8))
    _embed(np.zeros((MILO_SIZE,MILO_SIZE,3),dtype=np.uint8))


@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    return {
        "ok": True,
        "n": 0 if _vecs is None else int(_vecs.shape[0]),
        "identity": "tcgplayer",
        "detect": "yolo-tflite",
        "identify": "milo-cnn-128",
        "embedder": "milo_cnn",
        "orient": "90-270",
        "tcg_map": len(_tcg_to_ct),
        "catalogs": _catalogs.public_entries() if _catalogs else [],
        "version": "2.1-fast-worker",
        "worker": os.environ.get("CARDSCAN_WORKER", "local"),
        "device": "rocm" if any("ROCM" in p for p in _milo_providers) else "cpu",
        "milo_providers": list(_milo_providers),
    }


@app.get("/catalogs")
def catalogs():
    return {"ok": True, "catalogs": _catalogs.public_entries() if _catalogs else []}


def _box_reading_key(box: dict) -> tuple[float, float]:
    xyxy = box.get("xyxy") or [0, 0, 0, 0]
    return (float(xyxy[1] if len(xyxy) > 1 else 0), float(xyxy[0] if xyxy else 0))


def _hit_name_key(hit: dict) -> str:
    return "".join(ch for ch in str((hit or {}).get("name") or "").lower() if ch.isalnum())


def _photo_cards(payload: dict) -> list[dict]:
    cards = list(payload.get("cards") or [])
    cards.sort(key=lambda match: _box_reading_key(match.get("box") or {}))
    return cards


def _unique_top1_hits(matches: list[dict]) -> list[dict]:
    seen: set[tuple[str, str, str]] = set()
    hits: list[dict] = []
    for match in matches:
        top = match.get("top1") if isinstance(match, dict) else None
        if not top:
            continue
        key = (
            str(top.get("id") or ""),
            str(top.get("name") or ""),
            str(top.get("collector_number") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        hits.append(top)
    return hits


def _merge_album_payloads(payloads: list[dict]) -> dict:
    cards: list[dict] = []
    boxes: list[dict] = []
    detect_ms = 0.0
    identify_ms = 0.0
    embed_ms = 0.0
    search_ms = 0.0
    n_orient = 0
    for payload in payloads:
        boxes.extend(payload.get("boxes") or [])
        detect_ms += float(payload.get("detect_ms") or 0)
        identify_ms += float(payload.get("identify_ms") or 0)
        embed_ms += float(payload.get("embed_ms") or 0)
        search_ms += float(payload.get("search_ms") or 0)
        n_orient += int(payload.get("orientations") or 0)
    ranked = sorted(
        enumerate(payloads),
        key=lambda item: (-len(item[1].get("boxes") or []), item[0]),
    )
    _, primary = ranked[0] if ranked else (0, {})
    primary_boxes = len(primary.get("boxes") or [])
    dense = primary_boxes >= ALBUM_DENSE_BOX_COUNT
    seen_keys: set[tuple[str, str, str]] = set()
    seen_names: set[str] = set()
    unique: list[dict] = []

    def add_photo(payload: dict, extra: bool) -> None:
        photo_cards = _photo_cards(payload)
        extra_boxes = len(payload.get("boxes") or [])
        for match in photo_cards:
            cards.append(match)
            top = match.get("top1") if isinstance(match, dict) else None
            if not top:
                continue
            if float(top.get("score") or 0) <= ALBUM_UNIQUE_MIN_SCORE:
                continue
            key = (
                str(top.get("id") or ""),
                str(top.get("name") or ""),
                str(top.get("collector_number") or ""),
            )
            name_key = _hit_name_key(top)
            if key in seen_keys:
                continue
            if extra:
                if name_key and name_key in seen_names:
                    continue
                extra_score = float(top.get("score") or 0)
                # Dense binder + 1-box extras are sparse closeups (Boxed Order).
                # Sparse lots still union 1-box photos at 0.50. 2+ box extras
                # are album views and keep that same unique floor.
                if dense and extra_boxes <= 1 and extra_score < ALBUM_EXTRA_MIN_SCORE:
                    continue
            seen_keys.add(key)
            if name_key:
                seen_names.add(name_key)
            unique.append(top)

    if ranked:
        add_photo(primary, extra=False)
        for _idx, payload in ranked[1:]:
            add_photo(payload, extra=True)
    box_counts = [len(payload.get("boxes") or []) for payload in payloads]
    print(
        json.dumps({
            "cardscan": "identify-album",
            "photoCount": len(payloads),
            "boxCount": max(box_counts) if box_counts else 0,
            "uniqueCount": len(unique),
            "unique": [
                {
                    "name": hit.get("name"),
                    "collector_number": hit.get("collector_number"),
                    "score": hit.get("score"),
                    "card_back_score": hit.get("_card_back_score"),
                }
                for hit in unique
            ],
        }),
        flush=True,
    )
    return {
        "ok": True,
        "album": True,
        "identity": "public_id",
        "catalog": ALBUM_CATALOG,
        "game": "pokemon",
        "identify": "milo-cnn-128",
        "detect_ms": round(detect_ms, 1),
        "identify_ms": round(identify_ms, 1),
        "embed_ms": round(embed_ms, 1),
        "search_ms": round(search_ms, 1),
        "worker": os.environ.get("CARDSCAN_WORKER", "local"),
        "orientations": n_orient,
        "immediate": False,
        "boxes": boxes,
        "boxCount": max(box_counts) if box_counts else 0,
        "cards": cards,
        "hits": unique,
        "uniqueHits": unique,
        "albumMerged": True,
        "top1": unique[0] if unique else None,
        "photoCount": len(payloads),
    }


def _identify_locked_image(blob, top_k, live, catalog, multi, box_limit=None):
    if _vecs is None or _catalogs is None:
        raise HTTPException(503, "models not loaded")
    if catalog != "tcgplayer" and catalog not in _catalogs.entries:
        raise HTTPException(422, "unsupported catalog")
    identity = "tcgplayer" if catalog == "tcgplayer" else "public_id"
    game = "pokemon" if catalog == "tcgplayer" else _catalogs.entries[catalog]["game"]
    if not blob or len(blob) > MAX_BYTES:
        raise HTTPException(400, f"image must be 1–{MAX_BYTES} bytes")
    try:
        source = Image.open(BytesIO(blob))
        if source.width * source.height > 24_000_000:
            raise HTTPException(400, "image is too large; maximum 24 megapixels")
        im = ImageOps.exif_transpose(source).convert("RGB")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(400, "could not decode image") from exc
    w, h = im.size
    if max(w,h) > 1600:
        scale = 1600 / max(w,h)
        im = im.resize((max(1,round(w*scale)),max(1,round(h*scale))),Image.BILINEAR)
    rgb = np.asarray(im)
    t0 = time.perf_counter()
    boxes = _detect(rgb) if game == "pokemon" else card_quad.detect(rgb, live=live)
    # Catalog photos can already be cropped to their borders.
    if not boxes and not live and game == "pokemon":
        boxes = card_quad.detect(rgb, live=False)
    # Compare the full frame with detected crops for single gallery images.
    # This preserves border-to-border catalog photos without mistaking a
    # multi-card photo for one large card.
    if not live and not multi:
        ih, iw = rgb.shape[:2]
        if .60 <= min(iw,ih)/max(iw,ih) <= .80:
            boxes = [{"xyxy":[0.,0.,float(iw),float(ih)],"conf":1.,"detector":"card-frame"}] + boxes
    detect_ms = (time.perf_counter()-t0)*1000
    matches=[]; identify_ms=0.; n_orient=0
    timings={"embed_ms":0.,"search_ms":0.}
    limit = (4 if live else 8) if multi else (1 if live else 3)
    if game != "pokemon":
        limit = 8 if multi else 6
    if box_limit is not None:
        limit = max(1, min(int(box_limit), 36))
    if multi and game == "pokemon":
        boxes = sorted(boxes, key=_box_reading_key)
    for box in boxes[:limit]:
        crop = card_quad.crop(rgb,box)
        if not crop.size: continue
        t1=time.perf_counter()
        hits, tried = _embed_best(crop,top_k,live=live,catalog=catalog,timings=timings)
        identify_ms+=(time.perf_counter()-t1)*1000; n_orient+=tried
        # Drop western TCG backs so marketplace back photos are not returned as
        # Horsea/Froakie lookalikes. Uses the max card_back catalog score, not
        # only whether top1 was already rewritten to Pokemon Card Back.
        if _should_skip_card_back_box(hits, catalog):
            top = hits[0] if hits else {}
            print(
                json.dumps({
                    "cardscan": "skip-card-back",
                    "catalog": catalog,
                    "raw_top1": top.get("name"),
                    "raw_score": top.get("score"),
                    "card_back_score": top.get("_card_back_score"),
                }),
                flush=True,
            )
            continue
        matches.append({"box":box,"hits":hits,"top1":hits[0] if hits else None})
        if hits:
            print(
                json.dumps({
                    "cardscan": "identify-box",
                    "catalog": catalog,
                    "name": hits[0].get("name"),
                    "score": hits[0].get("score"),
                    "collector_number": hits[0].get("collector_number"),
                    "card_back_score": hits[0].get("_card_back_score"),
                }),
                flush=True,
            )
        # Restore the old Fast behavior: once a crop qualifies to open, stop.
        # Weak first proposals can still fall through to the refined borders.
        if live and not multi and _immediate(hits,[box],rgb.shape[1],rgb.shape[0]):
            break
    if not multi and matches:
        matches = [max(matches,key=lambda m:(m["top1"] or {}).get("score",-1))]
    if multi and game != "pokemon":
        # Compare nested outlines first, then keep the best recognition per card.
        distinct=[]
        for match in sorted(matches,key=lambda m:(m["top1"] or {}).get("score",-1),reverse=True):
            x1,y1,x2,y2=match["box"]["xyxy"]
            duplicate=False
            for prior in distinct:
                a,b,c,d=prior["box"]["xyxy"]
                overlap=max(0,min(x2,c)-max(x1,a))*max(0,min(y2,d)-max(y1,b))
                if overlap/max(1,min((x2-x1)*(y2-y1),(c-a)*(d-b))) > .65:
                    duplicate=True;break
            if not duplicate: distinct.append(match)
        matches=distinct
    if multi and game == "pokemon":
        matches.sort(key=lambda match: _box_reading_key(match.get("box") or {}))
    boxes = [m["box"] for m in matches]
    hits=matches[0]["hits"] if matches else []
    h,w=rgb.shape[:2]
    payload = {
        "ok":True,"identity":identity,"catalog":catalog,"game":game,
        "detect_ms":round(detect_ms,1),"identify_ms":round(identify_ms,1),
        "embed_ms":round(timings["embed_ms"],1),"search_ms":round(timings["search_ms"],1),
        "worker":os.environ.get("CARDSCAN_WORKER", "local"),
        "img_w":int(w),"img_h":int(h),"orientations":n_orient,
        "immediate":not multi and _immediate(hits,boxes,w,h),
        "boxes":boxes[:limit],"hits":hits,"top1":hits[0] if hits else None,
        "cards":matches,
    }
    if multi:
        payload["uniqueHits"] = _unique_top1_hits(matches)
    return payload


def _identify_image(blob, top_k, live, catalog, multi, box_limit=None):
    if _vecs is None or _catalogs is None:
        raise HTTPException(503, "models not loaded")
    if catalog != "tcgplayer" and catalog not in _catalogs.entries:
        raise HTTPException(422, "unsupported catalog")
    if not blob or len(blob) > MAX_BYTES:
        raise HTTPException(400, f"image must be 1–{MAX_BYTES} bytes")
    if not _lock.acquire(blocking=False):
        return {"ok": True, "busy": True, "catalog": catalog,
                "identity": "tcgplayer" if catalog == "tcgplayer" else "public_id"}
    try:
        return _identify_locked_image(blob, top_k, live, catalog, multi, box_limit=box_limit)
    finally:
        _lock.release()


def _identify_album_images(blobs, top_k, live):
    if _vecs is None or _catalogs is None:
        raise HTTPException(503, "models not loaded")
    if ALBUM_CATALOG not in _catalogs.entries:
        raise HTTPException(503, "album catalog not loaded")
    if not blobs:
        raise HTTPException(422, "file required")
    if len(blobs) > ALBUM_PHOTO_LIMIT:
        blobs = blobs[:ALBUM_PHOTO_LIMIT]
    for blob in blobs:
        if not blob or len(blob) > MAX_BYTES:
            raise HTTPException(400, f"image must be 1–{MAX_BYTES} bytes")
    if not _lock.acquire(timeout=ALBUM_LOCK_TIMEOUT_S):
        raise HTTPException(503, "identify busy")
    try:
        payloads = [
            _identify_locked_image(
                blob,
                top_k,
                live,
                ALBUM_CATALOG,
                True,
                box_limit=ALBUM_BOX_LIMIT,
            )
            for blob in blobs
        ]
        return _merge_album_payloads(payloads)
    finally:
        _lock.release()


_identify_hits = {}
_identify_hits_lock = threading.Lock()


# Per client IP, per minute. A phone's live camera loop sends ~2 frames/s
# (~120/min) for as long as it is open; at the old shared 30/min two of every
# three frames were refused and the phone's scan froze. Live frames get their
# own bucket so they never starve a photo/Add-card identify (live=0).
IDENTIFY_LIMIT_PER_MIN = 30
LIVE_IDENTIFY_LIMIT_PER_MIN = 240


def _limit_identify(request: Request, limit: int = IDENTIFY_LIMIT_PER_MIN, bucket: str = "photo") -> None:
    forwarded = request.headers.get("x-forwarded-for", "")
    ip = forwarded.split(",")[0].strip() if forwarded else ""
    if not ip:
        ip = request.client.host if request.client else "unknown"
    key = (ip, bucket)
    now = time.time()
    with _identify_hits_lock:
        hits = [stamp for stamp in _identify_hits.get(key, []) if now - stamp < 60]
        if len(hits) >= limit:
            _identify_hits[key] = hits
            raise HTTPException(status_code=429, detail="Too many identify requests.")
        hits.append(now)
        _identify_hits[key] = hits


@app.post("/identify")
async def identify(
    request: Request,
    file: UploadFile = File(...),
    top_k: int = Query(5, ge=1, le=8),
    live: bool = Query(False),
    catalog: str = Query("tcgplayer", max_length=64),
    multi: bool = Query(False),
    album: bool = Query(False),
):
    if live and not album:
        _limit_identify(request, LIVE_IDENTIFY_LIMIT_PER_MIN, "live")
    else:
        _limit_identify(request)
    blob = await file.read(MAX_BYTES + 1)
    if album:
        return await run_in_threadpool(_identify_album_images, [blob], top_k, live)
    return await run_in_threadpool(_identify_image,blob,top_k,live,catalog,multi)


@app.post("/identify-album")
async def identify_album(
    request: Request,
    file: list[UploadFile] = File(...),
    top_k: int = Query(1, ge=1, le=8),
    live: bool = Query(True),
):
    _limit_identify(request)
    blobs = []
    for upload in file[:ALBUM_PHOTO_LIMIT]:
        blobs.append(await upload.read(MAX_BYTES + 1))
    return await run_in_threadpool(_identify_album_images, blobs, top_k, live)


if WEB.is_dir():
    @app.get("/")
    def index():
        return FileResponse(WEB / "index.html", headers={"Cache-Control": "no-cache"})

    # Scan Connect: same page; web/static/scan-connect.js pairs with pokoin.com/inventory/scan.
    @app.get("/connect")
    def connect():
        return FileResponse(WEB / "index.html", headers={"Cache-Control": "no-cache"})

    if (WEB / "static").is_dir():
        app.mount("/static", StaticFiles(directory=WEB / "static"), name="static")
