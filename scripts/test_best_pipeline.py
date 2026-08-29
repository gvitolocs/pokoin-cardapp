#!/usr/bin/env python3
"""Accuracy + latency of the best Fast/Multi stack on the six GT photos.

Fast:  TCG YOLO .pt + Milo 128-d (no OCR). Identity = TCGplayer.
Multi: same YOLO crop → OpenCLIP 512-d vs CardTrader blueprint gallery.
       Rerank with Milo name/number tokens (free). 2-crop OCR only if gated.

Do not mix 268-d HOG, 128-d Milo, and 512-d CLIP as the same identity.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path("/home/nez/Projects/BattleScan")
PHOTOS = ROOT / "images"
TCG_PT = ROOT / "clones/TCG/backend/models/card_detector.pt"
INDEX = Path("/home/nez/Projects/pokoin/PokoinTest/index/blueprint_clip")
SCRIPTS = Path("/home/nez/Projects/pokoin/PokoinTest/scripts")
MIN_AREA = 0.01

GT_POKEMON = {
    "01-pikachu-sm86-slab.png": ["pikachu"],
    "02-mew-expedition-it-toploader.png": ["mew"],
    "03-dark-omanyte-neo-destiny.png": ["omanyte"],
    "04-deusolourdo-paldea-ir-fr.png": ["dudunsparce"],
    "05-slowpoke-delta-species-it.png": ["slowpoke"],
    "06-goldeen-pbl-087.png": ["goldeen"],
}
GT_BLUEPRINT = {
    "01-pikachu-sm86-slab.png": ["126853", "126854"],
    "02-mew-expedition-it-toploader.png": ["118584"],
    "03-dark-omanyte-neo-destiny.png": ["123435"],
    "04-deusolourdo-paldea-ir-fr.png": ["248881"],
    "05-slowpoke-delta-species-it.png": ["115063"],
    "06-goldeen-pbl-087.png": [],
}


def crop_box(img, xyxy):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return img[y1:y2, x1:x2]


def keep_box(img, xyxy) -> bool:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = xyxy
    return ((x2 - x1) * (y2 - y1)) / (w * h) >= MIN_AREA


def milo_hit(hit) -> dict:
    raw = dict(hit) if isinstance(hit, dict) else dict(hit.__dict__)
    meta = raw.get("metadata") or {}
    return {
        "name": raw.get("name"),
        "score": float(raw.get("score") or 0),
        "id": raw.get("id"),
        "collector_number": meta.get("collector_number"),
        "set": meta.get("set"),
    }


def should_ocr(clip_hits: list[dict], milo: dict | None) -> tuple[bool, str]:
    """OCR only when vision did not already pin a printing."""
    if not clip_hits:
        return True, "no_clip"
    names = [(h.get("name") or "").split()[0].lower() for h in clip_hits[:5]]
    same_species = len(set(names)) == 1
    gap = clip_hits[0]["cosine"] - clip_hits[1]["cosine"] if len(clip_hits) > 1 else 1.0
    milo_num = (milo or {}).get("collector_number") or ""
    if milo_num and any(h.get("bonus", 0) > 0 for h in clip_hits[:3]):
        return False, "milo_prior_hit"
    if same_species and gap < 0.04:
        return True, "print_tie"
    if gap < 0.02:
        return True, "low_gap"
    return False, "confident"


def main() -> None:
    sys.path.insert(0, str(ROOT / "clones/CollectorVision"))
    sys.path.insert(0, str(SCRIPTS))
    from ultralytics import YOLO
    import collector_vision as cvg
    import match_blueprint_clip as clipm
    import open_clip
    import torch
    from rapidocr_onnxruntime import RapidOCR

    files = [PHOTOS / n for n in GT_POKEMON]
    yolo = YOLO(str(TCG_PT))
    catalog = cvg.Catalog.load("pokemon")
    _ = catalog.embedder
    vecs, meta = clipm.load_index(INDEX)
    number_index = clipm.build_number_index(meta)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model = model.to(device).eval()
    ocr_engine = RapidOCR()
    print(f"clip_device={device} milo_n={len(catalog)} clip_n={len(meta)}", flush=True)

    rows = []
    for path in files:
        img = cv2.imread(str(path))
        t0 = time.perf_counter()
        result = yolo(img, verbose=False, conf=0.25, iou=0.45)[0]
        detect_ms = (time.perf_counter() - t0) * 1000
        boxes = []
        for b in result.boxes:
            xyxy = b.xyxy[0].cpu().numpy().tolist()
            if keep_box(img, xyxy):
                boxes.append({"xyxy": xyxy, "conf": float(b.conf[0])})
        boxes.sort(key=lambda x: x["conf"], reverse=True)
        crop = crop_box(img, boxes[0]["xyxy"]) if boxes else img
        pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

        t1 = time.perf_counter()
        emb = catalog.embedder.embed(pil)
        milo_hits = [milo_hit(h) for h in catalog.search_records(emb, top_k=5)]
        milo_ms = (time.perf_counter() - t1) * 1000
        milo0 = milo_hits[0] if milo_hits else None
        pokemon_ok = any(e in (milo0["name"] or "").lower() for e in GT_POKEMON[path.name]) if milo0 else False

        rgb = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        t2 = time.perf_counter()
        qn = clipm.encode(model, preprocess, device, rgb)
        clip_ms = (time.perf_counter() - t2) * 1000
        expect = GT_BLUEPRINT[path.name]

        clip_only = clipm.hits_from(vecs, meta, qn, 5, expect)
        milo_ocr = {
            "available": True,
            "text": " ".join(str(milo0.get(k) or "") for k in ("name", "collector_number", "set") if milo0),
            "numbers": clipm.parse_numbers(
                " ".join(str(milo0.get(k) or "") for k in ("name", "collector_number") if milo0)
            ),
            "name_tokens": clipm.tokens(milo0["name"] if milo0 else ""),
        }
        clip_milo = clipm.hits_from(
            vecs, meta, qn, 5, expect, ocr=milo_ocr, number_index=number_index
        )
        gate, gate_why = should_ocr(clip_milo["hits"], milo0)

        t3 = time.perf_counter()
        ocr2 = clipm.ocr_card(rgb, ocr_engine, extra=None)
        ocr2_ms = (time.perf_counter() - t3) * 1000
        clip_ocr2 = clipm.hits_from(
            vecs, meta, qn, 5, expect, ocr=ocr2, number_index=number_index
        )
        clip_gated = clip_ocr2 if gate else clip_milo

        rec = {
            "photo": path.name,
            "detect_ms": round(detect_ms, 1),
            "milo_ms": round(milo_ms, 1),
            "clip_encode_ms": round(clip_ms, 1),
            "ocr2_ms": round(ocr2_ms, 1),
            "n_boxes": len(boxes),
            "fast_top1": milo0,
            "fast_pokemon": pokemon_ok,
            "clip_only_top5": clip_only["correct_top5"],
            "clip_only_rank": clip_only["gt_ranks"],
            "clip_milo_top5": clip_milo["correct_top5"],
            "clip_milo_rank": clip_milo["gt_ranks"],
            "clip_ocr2_top5": clip_ocr2["correct_top5"],
            "ocr2_numbers": clip_ocr2.get("ocr_numbers"),
            "gate": gate,
            "gate_why": gate_why,
            "gated_top5": clip_gated["correct_top5"],
            "clip_top1": clip_only["hits"][0]["name"] if clip_only["hits"] else None,
            "milo_prior_top1": clip_milo["hits"][0]["name"] if clip_milo["hits"] else None,
        }
        rows.append(rec)
        print(
            json.dumps(
                {
                    "photo": path.name,
                    "fast": milo0["name"] if milo0 else None,
                    "clip_only": clip_only["correct_top5"],
                    "clip_milo": clip_milo["correct_top5"],
                    "ocr2": clip_ocr2["correct_top5"],
                    "gate": gate_why,
                    "gated": clip_gated["correct_top5"],
                    "ms": {
                        "detect": rec["detect_ms"],
                        "milo": rec["milo_ms"],
                        "clip": rec["clip_encode_ms"],
                        "ocr2": rec["ocr2_ms"],
                    },
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    known = [r for r in rows if GT_BLUEPRINT[r["photo"]]]
    summary = {
        "clip_device": device,
        "fast_pokemon": f"{sum(r['fast_pokemon'] for r in rows)}/{len(rows)}",
        "clip_only_top5": f"{sum(r['clip_only_top5'] is True for r in known)}/{len(known)}",
        "clip_milo_top5": f"{sum(r['clip_milo_top5'] is True for r in known)}/{len(known)}",
        "clip_ocr2_top5": f"{sum(r['clip_ocr2_top5'] is True for r in known)}/{len(known)}",
        "gated_top5": f"{sum(r['gated_top5'] is True for r in known)}/{len(known)}",
        "median_fast_ms": round(
            float(np.median([r["detect_ms"] + r["milo_ms"] for r in rows])), 1
        ),
        "ocr_calls": sum(1 for r in rows if r["gate"]),
        "rows": rows,
    }
    dest = ROOT / "results/bench-best-pipeline.json"
    dest.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps({k: summary[k] for k in summary if k != "rows"}), flush=True)
    print(f"-> {dest}", flush=True)


if __name__ == "__main__":
    main()
