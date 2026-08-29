# Pokoin scan stack — what we tried, what won, why

Measured on **nezopt** 2026-08-29 (Ryzen 5 8400F, 12 threads, RX 7900 XT/XTX, ROCm).
Ground truth: six BattleScan photos. Extra check: 50 Vinted listing photos (title-token overlap is a weak proxy, not blueprint GT).

Desktop scripts: `PokoinTest/scripts/scan_best.py`, `scan_fast.py`.
Benches: `BattleScan/results/bench-scan-steps.json`, `bench-opt-stack.json`, `bench-best-pipeline.json`, `bench-clip-gpu.json`, `bench-clip-fp16.json`, `vinted-clip-milo-prior.json`.

## Contract (do not mix)

| Stage | Job | Identity |
|---|---|---|
| Detect | Find the card box | pixels / quad |
| Fast identify | 1-NN in **128-d Milo** | TCGplayer product id |
| Multi identify | 1-NN in **512-d CLIP**, reranked by Milo **text** | CardTrader `blueprint_id` |
| HOG (APK fallback only) | 268-d native | LAYOUT |

Never add 268-d, 128-d, and 512-d into one score. Milo tokens on CLIP are metadata, not mixed embeddings.

---

## What won

### Fast (ship)

**TCG YOLO `.pt` detect + CollectorVision Milo 128-d ONNX + exact cosine.**

- Warm total **90–113 ms** (YOLO ~57 ms + Milo embed ~27–35 ms + search ~6–18 ms).
- Six GT: **6/6 species**, 5/6 printing.
- Printing miss: Expedition Mew **055** instead of holo **019** (glare / toploader; CLIP had the same miss).
- Pitch Black Goldeen **087/084**: Milo names it; CardTrader May dump has no single SKU.

CollectorVision does **not** use OCR. Neither does Fast.

### Multi (desktop)

**Same YOLO crop → OpenCLIP ViT-B-32 (full-card, not art-only) → 72 053×512 gallery, reranked with Milo name + collector number.**

- CLIP-only top-5: **2/5** (Omanyte, Slowpoke).
- CLIP + Milo prior: **5/5**. SM86 **rank 317 → 1**. Paldea Dudunsparce **rank 136 → 1**.
- RapidOCR is **off** the live path (see below).
- GPU encode ~66–77 ms + 7–10 ms search. Kernel-only fp16 **3.0 ms** vs fp32 **4.5 ms**.

### APK (this build)

Fast/Multi on device: **YOLO TFLite (TCG float16, 640 NHWC, `[1,5,8400]`) + Milo ONNX + bundled 27 027×128 catalog.**
CLIP encoder is still not packaged; Multi is **one Milo hit per YOLO box**. HOG → LAYOUT only if YOLO/Milo fail to load.

---

## Everything we tried

### Detect

| Method | Result | Why it lost / won |
|---|---|---|
| Canny / adaptive quad | 1–8 ms. Ruins CLIP on slabs (SM86 CLIP rank 775). Overlay-only is fine. | Fast enough, not accurate enough to crop for identify. |
| CollectorVision corners | Issue #24 rotated corners still open. We skip them. | YOLO already boxes the card. |
| Shrey YOLO | Vinted 48/50 boxes; missed sticker + generic listing. | TCG YOLO 50/50. |
| **TCG YOLO `.pt`** | Six GT: a box on every photo, conf ~0.89–0.98. Vinted 50/50. Warm **~40–59 ms** (median **56.8 ms** on Omanyte). | **Winner.** TCG deleted OpenCV detect after YOLO v2 mAP50-95 0.964. We use their floor: conf 0.25, IoU 0.45, min area 1%. |
| YOLO export ONNX + NMS | Ultralytics + online ORT guides say ONNX is often faster on CPU. | **115 ms vs 57 ms `.pt`.** Keep PyTorch on desktop. APK uses the **TFLite** TCG already ships (float16, 5.3 MB), not that ONNX. |
| YOLO INT8 TFLite | Same I/O layout, smaller file. | Not A/B’d on the six GT; float16 matches TCG mobile. Revisit if on-device YOLO is the new bottleneck. |

### Identify — Fast

| Method | Result | Why |
|---|---|---|
| Native HOG 268-d → LAYOUT | **0/6** LAYOUT on the six GT photos. | Gallery/descriptor mismatch vs phone photos of slabs/toploaders. Not the keeper. |
| Java 10584-d HOG | Explicitly banned (slow, not the native path). | Overlay stayed Canny. |
| **Milo 128-d ONNX** | 6/6 species, SM86 / 229/193 / 087/084 printings correct. Embed **~27 ms** at 4 threads. | **Winner.** Fine-tuned MobileViT-XXS ArcFace on card photos. Catalog n=27 027 pokemon. |
| CLIP 512-d as Fast | CLIP-only 2/5 print top-5; slower; identity is blueprint_id not TCGplayer. | Wrong identity space for Fast. |
| RapidOCR as Fast | 1.8–5.6 s. | Destroys the latency budget. CV does not OCR. |

### Identify — Multi (blueprint_id)

| Method | Six GT top-5 | Why |
|---|---|---|
| Canny warp → CLIP | 2/5. Ranks: SM86 775, Mew 2895, Dudunsparce 10798. | Query warp ≠ full-card CDN gallery. |
| YOLO crop → CLIP, no OCR | Still **2/5**, better ranks (SM86 317, Mew 75, Dudunsparce 136). JP Snow Hazard IR beat Paldea 248881 on photo 04. | Full-card crop is the right query. Print collisions remain. |
| Art-only crop 12–52% | TCG uses this because **their** CLIP is fine-tuned on art vs **their** gallery. | Our gallery is **full CDN cards**. Art queries domain-mismatch. Do not. |
| YOLO + RapidOCR 5-crop (name, collector, uncropped frame) | **5/5** top-5. SM86 via slab `sm86`. 229/193 via OCR. Mew 055 still first (OCR empty). | Accurate **and 5.6 s**. TCG Combined uses OCR because they fused it with a **fine-tuned** CLIP + RRF. We do not have that CLIP on Pokoin. |
| YOLO + RapidOCR 2-crop | **4/5.** **Missed SM86** (slab text is outside the YOLO crop). ~1.8 s. | Faster OCR, worse than Milo prior. |
| **YOLO + CLIP + Milo name/number prior** | **5/5**, **0 OCR calls**. SM86 and 229/193 rank 1. Mew 019 in top-5, 055 still top-1. | **Winner.** Milo already read SM86 / 229/193. Same token bonus OCR used, for free. |
| Qwen 3.8 as live rerank | Drafted the first OCR formula. Not on the live path. | Too slow; GPU is for CLIP/LoRA. Duty drafts only. |

### Search / runtime knobs (online advice vs this box)

| Knob | Advice | Measured | Choice |
|---|---|---|---|
| FAISS / IVFFlat | “Needed at 50–100k.” TCG uses IVFFlat because **pgvector in Postgres**, not because NumPy is slow. | 6 ms at 27k×128, 7–10 ms at 72k×512. | Exact cosine. Revisit above ~1M vectors. |
| `ORT_ENABLE_ALL` | ORT docs / blogs: turn all graph opts on. CV’s own `benchmark_onnx_models.py` sets it; `create_inference_session` does not. | Milo embed: **default 26.95 ms** vs ALL 31.33 ms (4 threads). | Leave CV default. |
| All CPU threads | Use the machine. | 4 threads **27 ms**, 12 threads **91 ms**. | **4 intra-op, 1 inter-op.** Oversubscription with YOLO+ORT+NumPy is real. |
| Reuse ORT session + warmup | Biggest ORT win. | CV already reuses one session. First YOLO ~150 ms, then ~50–80 ms. | Warm once at process start. |
| CLIP fp16 | TCG R2.3: ~1.8× on CUDA. | ROCm kernel **3.0 vs 4.5 ms**. Full encode still ~70 ms (decode + preprocess). | fp16 when Multi runs on ROCm. |
| `torch.inference_mode()` | TCG R2.4. | Cheap, correct. | Used in `match_blueprint_clip.encode`. |
| Preload models | TCG 1.1: first request 2–5 s. | CLIP load ~21 s on GPU, YOLO+Milo load ~0.75 s. | APK inits on a background thread at boot. |

### Vinted 50 (identify, title overlap)

| Engine | Overlap | Wall |
|---|---|---|
| CLIP cosine only | 22/50 | Cheap, weak print. |
| CLIP + RapidOCR | 33/50 | 313.7 s (~6.3 s/photo). OCR recovered some names and also leaked (Cinccino → Energy Pickup). |
| **CLIP + Milo prior** | **36/50** | **6.4 s** after model load (128 ms/photo). Ties CollectorVision. |
| CollectorVision Milo | 36/50 | Same overlap, TCGplayer ids not blueprint_id. |

Species agreement CV vs CLIP+OCR was 43/50. Real singles CLIP+OCR still lost: Charizard V SSR JP, Nymphli TG15, Pikachu 133/M-P, Cinccino. Piles/stickers are not a card-ID problem.

---

## Why we did **not** copy the other projects wholesale

**qtran/TCG** is the right detect (YOLO) and the right “OCR is a print tool” lesson. Their Combined mode (OCR weight 2, image 1, RRF, art crop, fine-tuned CLIP, pgvector IVFFlat) is built for **their** gallery and **their** fine-tune. Pointing TCG art-CLIP at Pokoin full-card CDN is the domain gap that made CLIP-only 2/5.

**CollectorVision** is the right Fast identify (Milo, no OCR, brute cosine). We do **not** use their corner detector.

**Shrey** is EasyOCR vs CLIP filenames. Weaker detect than TCG; not the keeper.

**TrainingAI** wanted Tesseract + Meili. tesseract was not installed; RapidOCR was the stand-in. After measurement, RapidOCR is not live Fast/Multi.

---

## Remaining holes (re-examine if…)

1. **Expedition Mew 019 vs 055** — both engines. Would re-examine with a holofoil-aware embedder or TCG’s Combined **if** we fine-tune CLIP on Pokoin full-card pairs (TCG v9), not before.
2. **Pitch Black singles** missing from May `best-blueprint-images.json`. Regen catalog; Fast already names 087/084.
3. **CLIP encoder on APK** — TCG has `clip_visual.tflite`. 72k×512 is ~147 MB. Ship later; Multi today is YOLO+Milo per box.
4. **On-device YOLO latency** — desktop 57 ms PyTorch CPU. Phone TFLite+NNAPI not benched yet. If live preview is janky, drop to INT8 or run detect every Nth frame (CameraX already drops busy frames).
5. **blueprint_id ↔ LAYOUT bijection** — still none. Do not pretend LAYOUT is CardTrader.

---

## Why the losing options looked attractive

(Qwen 3.8, 2026-08-29, numbers taken only from the benches above.)

Every rejected knob had a plausible default from docs or another project. On this box the profiler disagreed.

**ONNX instead of PyTorch.** Integration guides say export to ONNX and you will be faster. TCG YOLO was **56.8 ms** as `.pt` and **115 ms** as ONNX. The graph looked clean. It was not faster here, so desktop Fast keeps Ultralytics `.pt`. The APK uses TCG’s TFLite, not that ONNX.

**`ORT_ENABLE_ALL`.** The ORT docs recommend it. CollectorVision’s bench script sets it; their library session does not. Milo at 4 threads: **31 ms** with ALL, **27 ms** default. We kept default.

**Twelve threads.** More cores looks like more speed. **12 threads → 91 ms**, **4 threads → 27 ms**. Oversubscription with YOLO + ORT + NumPy on a 6-core CPU.

**FAISS / IVFFlat.** Search is already **6–10 ms** at 27k and 72k. TCG uses IVFFlat because the vectors live in **Postgres**, not because NumPy is slow.

**RapidOCR.** Five crops: **5.6 s**, 5/5 GT. Two crops: **1.8 s**, missed SM86. Milo’s name/number prior: **5/5** with no extra model. OCR *worked*; it was not the live path.

**Art crop 12–52%.** Correct for TCG’s fine-tuned CLIP against TCG’s gallery. Wrong for Pokoin’s full-card CDN gallery.

**HOG 268-d.** Classic layout descriptor. **0/6 LAYOUT** on these photos.

**CLIP alone.** 2/5 GT, 22/50 Vinted. CLIP+OCR 33/50 in 314 s. CLIP+Milo tokens **5/5** and **36/50 in 6.4 s**.

---

## How to run

Desktop Fast+Multi:

```bash
/home/nez/Projects/BattleScan/.venv/bin/python \
  /home/nez/Projects/pokoin/PokoinTest/scripts/scan_best.py --mode both \
  /home/nez/Projects/BattleScan/images/0{1,2,3,4,5,6}-*.png
```

APK identify assets:

```bash
/home/nez/Projects/BattleScan/.venv/bin/python \
  /home/nez/Projects/BattleScan/scripts/export_apk_identify_assets.py
cd /home/nez/Projects/pokoin/PokoinTest && ./prepare-apk-assets.sh
cd android && ./gradlew assembleDebug
```
