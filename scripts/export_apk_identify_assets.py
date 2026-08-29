#!/usr/bin/env python3
"""Pack YOLO TFLite + Milo ONNX + pokemon catalog for the debug APK."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "android/app/src/main"
TFLITE = ROOT / "clones/TCG/backend/models/card_detector_float16.tflite"
MILO = ROOT / "clones/CollectorVision/collector_vision/weights/milo.onnx"


def main() -> None:
    sys.path.insert(0, str(ROOT / "clones/CollectorVision"))
    import collector_vision as cvg

    models = APP / "assets/models"
    idx = APP / "assets/milo_index"
    models.mkdir(parents=True, exist_ok=True)
    idx.mkdir(parents=True, exist_ok=True)
    shutil.copy2(TFLITE, models / "card_detector.tflite")
    shutil.copy2(MILO, models / "milo.onnx")

    catalog = cvg.Catalog.load("pokemon")
    vecs = np.asarray(catalog.embeddings, dtype=np.float32)
    nrm = np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-8, None)
    vecs = vecs / nrm
    (idx / "embeddings.bin").write_bytes(np.ascontiguousarray(vecs).tobytes())
    lines = []
    for i in range(len(catalog)):
        rec = catalog.record_for_index(i)
        meta = rec.get("metadata") or {}
        lines.append(
            json.dumps(
                {
                    "id": rec.get("id"),
                    "name": rec.get("name"),
                    "n": meta.get("collector_number"),
                    "set": meta.get("set"),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
    (idx / "metadata.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "n": int(vecs.shape[0]),
        "dim": int(vecs.shape[1]),
        "dtype": "float32",
        "yolo": "card_detector_float16.tflite",
        "milo": "milo.onnx",
        "identity": "tcgplayer",
    }
    (idx / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        json.dumps(
            {
                "n": manifest["n"],
                "emb_mb": round((idx / "embeddings.bin").stat().st_size / 1e6, 2),
                "meta_mb": round((idx / "metadata.jsonl").stat().st_size / 1e6, 2),
                "tflite_mb": round((models / "card_detector.tflite").stat().st_size / 1e6, 2),
                "milo_mb": round((models / "milo.onnx").stat().st_size / 1e6, 2),
            }
        )
    )


if __name__ == "__main__":
    main()
