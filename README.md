# pokoin-cardapp

Pokoin on-device card scan. This replaces the Flutter TrainingAI client (`SCAN_CARD_IA`).

| Path | Identity | What it does |
|---|---|---|
| **Fast** | TCGplayer | TCG YOLO detect + CollectorVision Milo 128-d |
| **Multi** (desktop) | CardTrader `blueprint_id` | Same YOLO crop → OpenCLIP ViT-B-32, reranked with Milo name/number tokens |
| **APK** | TCGplayer | YOLO TFLite + Milo ONNX + bundled 27k catalog. CLIP encoder is not packaged. |

Do not mix HOG 268-d, Milo 128-d, and CLIP 512-d. RapidOCR is off the live path.

Measured numbers, rejected knobs, and why: [`notes/SCAN_STACK.md`](notes/SCAN_STACK.md).

## Android

Models and the Milo catalog are **not** in git. After cloning the three keeper repos:

```bash
bash scripts/clone-scanners.sh
python scripts/export_apk_identify_assets.py
cd android && ./gradlew assembleDebug
```

## Desktop bench

```bash
python -m venv .venv
. .venv/bin/activate
bash scripts/clone-scanners.sh
python scripts/run_keepers.py images/YOUR.jpg
```

Drop photos in `images/`. Do not commit them.

Re-clone keepers: `bash scripts/clone-scanners.sh`
