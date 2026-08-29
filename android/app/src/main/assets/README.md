# Generated identify assets

Not committed. After `bash scripts/clone-scanners.sh`:

```bash
python scripts/export_apk_identify_assets.py
```

That writes:

- `models/card_detector.tflite` — TCG YOLO float16
- `models/milo.onnx` — CollectorVision Milo
- `milo_index/embeddings.bin` + `metadata.jsonl` — pokemon catalog, 128-d, TCGplayer ids
