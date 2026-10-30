# Rust worker vs Python worker (2026-10-29)

Same index (`PokoinTest/index/cdn_milo_2026-10-29`, 75,146 rows), same Milo ONNX, same YOLO11n ONNX detector,
4 ORT threads each, both on nezopt. A = `server/cdn_app.py` with `CARDSCAN_DETECTOR=onnx` (:18098),
B = `cardscan-worker` (:18096). Requests interleaved A/B, one at a time, with `scan-compare` (curl, client side).
nezopt was busy with other sessions (load 40–70, Tctl 84–93 °C); runs were gated to start below 90 °C.
Raw reports: nezopt `~/scratch/cardscan-refresh-2026-10-29/py-vs-rust-real100*.json`.

Images: 100 real photos (`BattleScan/images/vinted` 50, `images/*.png` 6, `images/vinted_condition` 44).

## Parity (no `catalog`, both search all rows)

- Top-1 identical on **98/100**. The 2 differences are near-ties on non-card photos (merchandise):
  0.5264 vs 0.5263 and 0.5415 vs 0.5411.
- Score difference on agreeing top-1: mean 0.0015, max 0.024 (JPEG decoder and resize rounding differ slightly
  from Pillow/libjpeg-turbo).
- Numel (`370861_numel.jpg`, `catalog=pokemon_western`): top-1 370861 "Numel", Ascended Heroes, 0.9973.

## Latency (200 requests each, 2 reps × 100 images, ms)

| | mean | p50 | p90 | p99 | max |
| --- | --- | --- | --- | --- | --- |
| Python + ONNX detector | 258 | 227 | 463 | 678 | 822 |
| **Rust** | **167** | **138** | **271** | **584** | 852 |

Server-reported stages, mean / p50: detect Python 50 / 38, Rust 55 / 42; identify (Milo × orientations + search)
Python 196 / 176, Rust 101 / 82.
For reference, production (Python + PyTorch detector) took 7–38 s per scan tonight under the same load;
the 2026-10-28 study measured mean 627 ms, p90 1.8 s.

Memory (RSS after the run): Rust 359 MB, Python + ONNX 383 MB, production Python + PyTorch 673 MB.

## Catalog routing (`catalog=pokemon_western`; Python ignores it, Rust searches western + uncatalogued rows)

30/100 top-1 change. On Vinted photos whose listing title names the card, routing fixes western cards that the
unrouted search answered with a Japanese or Chinese printing of the same art:
Gengar POR 050, Hop's Wooloo JTG 170, Kabutops 7/92, Jirachi 97/95, Roserade DP 13/130, Bulbasaur 133/132,
Wugtrio 190/162, Punk Helmet 121/094, Cinccino ex (unrouted: Ortega).
Cards titled Japanese (m2a 244/193, 243/193, 233/193, s4a, sm11b, 133/M-P, 208/S-P) become the western printing when
the client asks for `pokemon_western`, as that parameter requests; Yu Nagaba Pikachu 208/S-P has no western
printing and falls to a wrong card (Flittle 0.56).

## Not checked

- Production-identical comparison with the PyTorch detector (the 2026-10-28 study found ONNX vs PyTorch same card 28/30).
- Concurrency behaviour under parallel load (both serialize on one model lock by design).
- iPhone/Android clients against the Rust worker; the Pi API adds its own fields to some responses.
