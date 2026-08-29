# How to feed one photo

```bash
cd /home/nez/Projects/BattleScan
. .venv/bin/activate
python scripts/run_keepers.py images/YOUR.jpg
```

That times:

1. **qtran TCG YOLO** — `clones/TCG/backend/models/card_detector.pt`
2. **Shrey YOLO** — `clones/Pokemon-Card-Scanning-Webapp/detector_models/pokemon_detector4/weights/best.pt`
3. **CollectorVision** — Milo catalog search on the highest-confidence TCG box crop

`results/<stem>/timing.json` has cold vs warm milliseconds. Warm = median of 5 runs after 1 warmup.

Do not commit photos.
