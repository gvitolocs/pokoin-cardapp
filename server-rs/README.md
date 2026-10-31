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

* absent, empty, `all` or `pokemon` → search every row with a non-zero vector, response `"catalog":"all"`;
* `pokemon_western`, `pokemon_japanese`, `pokemon_chinese` → rows whose metadata
  `catalogs` contains that id **plus rows with empty `catalogs`** (uncatalogued
  cards stay findable), response `"catalog":"<id>"`;
* anything else → every row, `"catalog":"all"`, and a warning on stderr.

The per-catalog row lists are built once at load using `cardscan_core::route_rows`, then filtered to exclude zero vectors. `/health` reports all metadata rows in `n` and zero-vector rows in `no_image`.

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

## Index refresh and placeholder pictures

Some CDN faces are not the card: CardTrader's grey card back (186x260, `fallbacks/card_uploader/preview.png`) and
Pokoin's coin-mascot missing-card art. Embedded, they become one vector shared by hundreds of rows, and a blurry
or grey photo then "matches" a placeholder. `card-placeholder` recognises both (size rule, difference hash and grey
thumbnail against `card-placeholder/refs/`). Every step that feeds the index uses it:

- `card-images audit` lists placeholder faces across the objects tree (all games). `--previous audit.jsonl`
  uses that file's mtime as the cutoff: unchanged placeholder rows are carried over, disappeared rows drop
  out, and newer faces are examined. Preserve the snapshot mtime with `cp -p`. Reports include `examined`
  and `carried_over`; per-game counts cover examined files plus carried placeholders, not skipped real files.
  Header-size checks still apply. New shared-byte groups can only be discovered among examined files and
  carried placeholders; older real files are not rehashed. Use a full audit periodically to rediscover those groups.
- `card-images recover` finds the real picture (same-blueprint sibling file, CardTrader full-size upload, TCGdex,
  pokemontcg.io, Scryfall), checks name, number and that it is a real card, never uses a base-set scan for a
  stamped printing, and with `--apply` writes it under the same key (originals backed up, `_homepage.webp` and
  `previews/` regenerated). `recovered.jsonl` records source and licence per picture; `missing.jsonl` is what has
  no picture.
- `index-refresh mark --index DIR --objects DIR --out DIR [--cos 0.995] [--threads N]` keeps every row
  in order, replacing placeholder embeddings with all-zero vectors and setting `"image": "none"`. It also marks
  stored vectors near a placeholder face's embedding, including files since recovered. Already-zero rows stay
  no-image; undecodable faces (such as AVIF) are retained unless their vector matches a placeholder.
  `--out` must not exist. `mark-report.json` records rows, newly marked counts by reason, and total `no_image`;
  `progress.json` records completion.
- `index-refresh plan` stages new real faces in `faces/` with ids in `allow.txt`, and recovered faces for existing
  zero-vector rows with ids in `reembed.txt`. Missing cards with unusable faces go in `noimage.jsonl` with full
  metadata and `"image": "none"`; `report.json` counts each group.
- `index-refresh purge` is a **legacy**, destructive-to-row-count export command: it drops placeholder rows.
  It remains available but is no longer used by the refresh pipeline. `card-images catalog-purge` is also legacy.
- `card-images catalog-mark --catalogs DIR --audit FILE --out DIR [--cos 0.995] [--only ids] [--embeddings FILE]`
  keeps app rows and their field order, zeroes audited/near-placeholder f16 vectors, and merges sorted public
  ids into each catalog's `noImagePublicIDs`. Existing zero rows stay no-image. Unchanged artifacts are hard-linked;
  updated artifacts get content-hashed names. Output must be new and outside the source. Packs with a manifest's
  `catalog_sha256` matching the catalog pin it against rewriting. The index preserves other keys and key order;
  `placeholderMarkExport` records provenance and per-catalog counts (including pinned skips).
  Embedding JSONL `{catalog,id,vector:[128 floats]}` replaces a public id's vector and clears no-image status.
  An optional string `image` updates the existing image cell when recovering a row appended without a face.
  An `append:{id,name,number,set,image}` object adds an absent public id in the existing field order, with other
  cells null. A null vector appends a no-image row. Repeated appends are ignored. Cells remain strings or null.

`scripts/refresh-index.sh BASE NEW WORK` runs audit → recover (`APPLY=1` to write) → plan → embed → enrich → mark guard.
Use `MARK_BASE=1` on the first run with an old index that still contains placeholder vectors: it marks BASE into
`WORK/base-marked` before planning. The guard fails if any non-zero row is newly marked; existing no-image rows
are fine. Guard artifacts remain in `WORK/guard` for inspection. `SKIP_AUDIT=1 AUDIT=FILE` shares the index
stages with the daily driver. If allow/reembed/noimage are all empty, it leaves NEW uncreated.

The Python Milo bridge accepts `scripts/embed_append.py BASE FACES ALLOW OUT --reembed REEMBED --noimage NOIMAGE`.
It checks embedder parity, replaces the requested zero-vector rows in place (preserving metadata except `image`,
which becomes the staged face filename), appends allowed real faces, then appends new no-image metadata with zero
vectors, skipping ids already present. Its progress includes `replaced` and `noimage_appended`. The index format
remains f32 `[N,128]` `embeddings.npy` plus `metadata.jsonl`; no-image cards remain real index rows but cannot match.
The image gallery is the parity reference (`--gallery DIR` is repeatable, in lookup order); after the index is
written, re-embedded faces replace gallery pictures with backups, and appended faces are added to the last gallery directory.

The existing `pokoin-all-cardtrader-daily` nightly job (23:00 UTC) should call
`~/.local/lib/card-refresh/current/daily-card-refresh.sh` after its daily export. This repository supplies the
driver; wiring the nightly job is maintained in `gvitolocs/pokoin`. Install with `scripts/install-card-refresh.sh`:
it builds the two Rust tools, copies scripts and binaries into a git revision directory, and atomically selects
`current`. Repository execution uses `target/release`; installed execution finds everything beside the driver.

The driver locks `$CARD_STATE/lock` (`CARD_STATE=/home/nez/data/card-images` by default), preserves
`audit.prev.jsonl`, audits incrementally, recovers with backups, and audits again before plan/embed/enrich/guard.
Recovery uses the latest daily export and the datasets in `card-refresh-common.sh`. `EXPANSIONS` can provide a
TSV instead of reading Postgres. `BASE_INDEX`, `OBJECTS`, `CATALOGS`, `DAILY`, `TCG_DATASETS`, and `EMBED_PYTHON`
override inputs. CPU-heavy work uses a 400% systemd scope and waits when Tctl reaches 90 °C.

With `SWAP=1` (default), a changed, guarded index gets `60-index.conf`, preserving previous matching drop-ins
in `runs/<UTC timestamp>/dropins`. Health must report the new basename and row count; a direct `/identify?top_k=1`
multipart `file` request must identify Numel ct_id 370861. Failures restore the drop-ins and restart the previous
worker. `SWAP=0` stages without switching. Only indexes recorded by this driver in `indexes.txt` are eligible
for retention deletion; the last five and the current/previous indexes are retained.

`scripts/app_embed.py` imports the scanner's `card_cnn` preprocessing and runs the catalog MiloCNN on CPU
(batch 8, four threads). It verifies 30 random western rows against stored f16 vectors, requiring minimum cosine
0.995 (exit 2 on parity failure), then emits recovered no-image vectors and new cards from plan's cardmeta.
Western, Japanese, and Chinese catalogs are eligible; pack-pinned catalogs are skipped. Missing faces append
with an empty image and zero vector. The driver stages changed catalogs as `<YYYYMMDD>-auto-cards` (with suffixes)
under `/home/nez/data/cardrails-catalogs`; `current` is repointed only with `CARDRAILS_CATALOG_AUTOPUBLISH=1`
(default off). It prints the staged path otherwise.

`DRY_RUN=1 CARD_STATE=/path/to/scratch scripts/daily-card-refresh.sh` downloads recovery candidates only into
the run directory, never applies recovery, swaps, publishes, or prunes live indexes. Index and catalog outputs
go under `$CARD_STATE/dry/`. Live index discovery uses read-only `/health` in dry mode; it makes no `systemctl`
calls. Each run writes logs and `summary.json` below `runs/<UTC timestamp>/`, and prints a compact summary line
with audit counts, recovery sources, missing reasons, index rows/no-image count, swap and catalog status.

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
# live index set by drop-in nezopt-cardscan-rs.service.d/60-index-2026-10-30.conf: cdn_milo_2026-10-30b
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
