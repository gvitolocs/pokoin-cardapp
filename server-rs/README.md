# server-rs

Rust tooling for the BattleScan / Pokoin scanner.

| crate | what it is |
|---|---|
| `index-refresh` | index tooling (missing-card diff, staging, metadata enrich) |
| `scan-compare` | A/B HTTP comparison of two `/identify` servers |
| `cardscan-core` | pure decision rules, Verus-verified where the rule is integer/fixed-point |
| `cardscan-worker` | the HTTP worker: the Rust port of `server/cdn_app.py` |

`cardscan-worker` implements the same HTTP contract as `server/cdn_app.py`
(`GET|HEAD /`, `GET|HEAD /health`, `POST /identify`) and adds the `catalog`
query parameter. It is CPU-only (ONNX Runtime, no PyTorch, no Python).

## Build

The worker uses `ort` with the `load-dynamic` feature, so it needs an
`libonnxruntime.so` at run time. The one that ships with the BattleScan Python
venv is ONNX Runtime **1.29.0**:

```sh
export ORT_DYLIB_PATH=/home/nez/Projects/Startup/Pokoin/BattleScan/.venv/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so.1.29.0
```

`ort` 2.0.0-rc.10 targets the ONNX Runtime C API version 22 (ORT 1.22).
ONNX Runtime keeps its C ABI backwards compatible, so 1.29.0 serves the 1.22
API and works unchanged. Anything >= 1.22 is fine.

```sh
cd server-rs
cargo build --release -p cardscan-worker
```

The release profile is the workspace one (`lto = "thin"`, `codegen-units = 1`).

## Run

```sh
export ORT_DYLIB_PATH=/home/nez/Projects/Startup/Pokoin/BattleScan/.venv/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so.1.29.0
export CDN_MILO=/home/nez/Projects/Startup/Pokoin/PokoinTest/index/cdn_milo_2026-10-29
export MILO_ONNX=/home/nez/Projects/Startup/Pokoin/BattleScan/clones/CollectorVision/collector_vision/weights/milo.onnx
export CARDSCAN_WEB=/home/nez/Projects/Startup/Pokoin/BattleScan/web
./target/release/cardscan-worker
```

| env | default | meaning |
|---|---|---|
| `CARDSCAN_BIND` | `127.0.0.1` | listen address |
| `CARDSCAN_PORT` | `8096` | listen port |
| `CARDSCAN_THREADS` | `4` | ONNX Runtime intra-op threads (inter-op is 1) |
| `CARDSCAN_WEB` | `web` | directory containing `index.html` |
| `CDN_MILO` | `/home/nez/Projects/Startup/Pokoin/PokoinTest/index/cdn_milo` | index dir: `embeddings.npy` (`f32 [N,128]`) + `metadata.jsonl` |
| `MILO_ONNX` | BattleScan `clones/CollectorVision/collector_vision/weights/milo.onnx` | Milo embedder |
| `YOLO_ONNX` | BattleScan `clones/TCG/backend/models/card_detector.onnx` | YOLO11n detector with built-in NMS; keep its external `.onnx.data` next to it |
| `ORT_DYLIB_PATH` | — | ONNX Runtime shared library (required by `load-dynamic`) |

The index is loaded and L2-normalized at startup. Startup fails if the
metadata row count (after skipping non-digit `ct_id`s) differs from the
embedding rows, matching `_load` in `cdn_app.py`.

## Catalog routing

`POST /identify?...&catalog=<id>`:

* absent, empty, `all` or `pokemon` → search every row, response `"catalog":"all"`;
* `pokemon_western`, `pokemon_japanese`, `pokemon_chinese` → rows whose metadata
  `catalogs` contains that id **plus rows with empty `catalogs`** (uncatalogued
  cards stay findable), response `"catalog":"<id>"`;
* anything else → every row, `"catalog":"all"`, and a warning on stderr.

The per-catalog row lists are built once at load by `cardscan_core::route_rows`.

## Tests and checks

```sh
cd server-rs
cargo test --workspace
cargo clippy --workspace -- -D warnings

# model-dependent integration test (Numel photo -> ct_id 370861):
ORT_DYLIB_PATH=... CDN_MILO=... MILO_ONNX=... \
  cargo test -p cardscan-worker --test integration -- --ignored --nocapture

# Verus (one verification run at a time on nezopt):
export PATH=$HOME/tools/verus/verus-x86-linux:$PATH
~/.local/bin/verify-gate -- cargo verus verify -p cardscan-core
```

Verus proves the integer/fixed-point rules in `cardscan-core`: top-k bounds
(1..=8, live <= 3) and sorted selection, orientation kind/count (plan length
1/2/4), early-exit and best-replacement, immediate, crop tenths/clamping,
the long-edge integer truncation, the letterbox inverse clamp
(`[0, size]`), and that every routed row index is `< N`. The `f32`/`f64`
adapters (`inverse_tenths`, `letterbox`, `crop_rect`, `score_key`, ...) are
outside `verus!` and covered by unit tests. Under plain `cargo` the specs are
erased, so the shipped code is the verified code.

## Smoke test

```sh
ORT_DYLIB_PATH=... CDN_MILO=... MILO_ONNX=... CARDSCAN_PORT=18096 \
  ./target/release/cardscan-worker &
curl -s -F file=@/home/nez/data/pokoin-leftovers/objects/370861_numel.jpg \
  'http://127.0.0.1:18096/identify?catalog=pokemon_western&top_k=3'
./target/release/scan-compare --a http://127.0.0.1:8098 --b http://127.0.0.1:18096 \
  --images /home/nez/scratch/cardscan-refresh-2026-10-29/known30.txt
```

## Production on nezopt (since 2026-10-30)

`nezopt-cardscan-rs.service` (user unit) serves 127.0.0.1:8098, the port the Pi's SSH tunnel reaches, in place of
the Python `nezopt-cardscan.service`. The two units conflict: starting one stops the other.

- Binary: `~/.local/lib/cardscan-rs/<commit>/cardscan-worker`, with `current` pointing at the live one, so the worktree can be
  rebuilt or removed without touching production. To deploy a new build: copy it to a new `<commit>` directory, move
  `current`, then `systemctl --user restart nezopt-cardscan-rs`.
- Rollback to the Python worker (same index): `cardscan-rollback` (`~/.local/bin`), which disables the Rust unit and
  enables and starts the Python one.

```ini
[Unit]
Description=Pokoin cardscan identify on nezopt (Rust worker, CDN-Milo, ONNX detector)
After=network-online.target
Wants=network-online.target
Conflicts=nezopt-cardscan.service

[Service]
Type=simple
Environment=HOME=/home/nez
Environment=ORT_DYLIB_PATH=/home/nez/Projects/Startup/Pokoin/BattleScan/.venv/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so.1.29.0
Environment=CDN_MILO=/home/nez/Projects/Startup/Pokoin/PokoinTest/index/cdn_milo_2026-10-29
Environment=MILO_ONNX=/home/nez/Projects/Startup/Pokoin/BattleScan/clones/CollectorVision/collector_vision/weights/milo.onnx
Environment=YOLO_ONNX=/home/nez/Projects/Startup/Pokoin/BattleScan/clones/TCG/backend/models/card_detector.onnx
Environment=CARDSCAN_WEB=/home/nez/Projects/Startup/Pokoin/BattleScan/web
Environment=CARDSCAN_THREADS=4
Environment=CARDSCAN_BIND=127.0.0.1
Environment=CARDSCAN_PORT=8098
ExecStart=/home/nez/.local/lib/cardscan-rs/current/cardscan-worker
Restart=always
RestartSec=5
MemoryMax=4G

[Install]
WantedBy=default.target
```
