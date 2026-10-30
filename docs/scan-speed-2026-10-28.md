# Scanner and CardRails speed study (2026-10-28)

Source: the YouTube Short "I challenged Rust Developers" (Arjay McCandless). An Axum + SQLite API on a $12 VM
went from 14k to 64k concurrent users by dropping nginx/framework/runtime/ORM for a hand-written io_uring server,
multishot accept/recv, 64-byte connection records with buffer reuse, batched SQLite work and a source-built SQLite.
The real ceiling then was the kernel's conntrack table, not the code.
The lesson that transfers: **measure where the time goes, drop heavy layers on the hot path, batch, don't copy.**

## 1. Card scanner (priority)

**What actually runs.** `api.pokoin.com/api/scan` → Pi (Rust API) → SSH reverse tunnel → nezopt
`nezopt-cardscan.service` = `BattleScan/server/cdn_app.py` (FastAPI/uvicorn, CPU only, 4 ORT threads):

1. PIL decode
2. resize to ≤ 1600 px
3. YOLO11n card detector through **ultralytics + PyTorch** (640 px)
4. crop
5. Milo ONNX embedding (448 px, 128-d), 1–4 orientations in sequence, early exit at 0.82
6. brute-force dot product over 74,516 catalog vectors

A global lock serializes requests.
Note: `BattleScan/server/` is **not in git** (untracked in the main checkout).

**Profile** (30 real photos, Vinted listings and reference shots, 2–3 passes; nezopt was busy at Tctl 87–91 °C, so the
tails are realistic for this host):

| Stage | p50 | p90 | Note |
| --- | --- | --- | --- |
| JPEG decode + resize | 25 + 1 ms | 67 + 40 ms | mean upload 612 kB |
| YOLO (ultralytics/PyTorch) | 34–112 ms | 295–1,754 ms | **the tail**: PyTorch's 6 intra-op threads + ORT's 4 oversubscribe the CPU |
| Milo per orientation | 39 ms (+9 ms preprocessing) | 92 ms | 1.6–2 orientations per scan; model has fixed batch 1 |
| Vector search (74k × 128 f32) | 14 ms | 45 ms | memory-bound matvec |
| **Server total** | **~280–314 ms** | **~1.8–2.0 s** | |
| Pi → tunnel → nezopt hop | +22 ms per request | | LAN ping is 0.16 ms |

### Video techniques mapped to the scanner, ranked

| # | Technique (video) | Where | Measured / expected gain | Risk | Effort | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Drop the heavy framework on the hot path (no Axum/Tokio/sqlx → here no ultralytics/PyTorch) | `_detect` → ONNX Runtime on the existing `card_detector.onnx` (YOLO11n with NMS built in) | **End to end: mean 627 → 181 ms (3.5×), p50 280 → 179, p90 1,812 → 273, p99 6,244 → 377 ms**; PyTorch no longer loaded (less RAM, faster start) | same card 28/30; the 2 differences are near-ties (0.54/0.54, 0.72/0.73) from a 1–2 % box shift | S (prototype done) | **Do** |
| 2 | Know your real bottleneck (conntrack lesson) | thread oversubscription; cap PyTorch threads if the pt path stays | mean 627 → 430, p90 1,812 → 500 ms; same card 30/30 | none | XS | Do (only if #1 is not taken) |
| 3 | Avoid copies/allocations | JPEG `draft()` (decode at reduced DCT scale), drop the 1600 px resize + `[:, :, ::-1].copy()` | decode p90 35 → 23 ms; same card 30/30 | none | XS | Do (in the prototype) |
| 4 | Batching | Milo re-exported with a dynamic batch so 2–4 orientations run in one call; or pick the orientation from the box aspect first | ~20–35 ms per two-orientation scan | re-export must match embeddings exactly | M | Maybe |
| 5 | Smaller records / less memory traffic | catalog matrix f32 → f16/int8 (38 MB → 19/9.5 MB) | search 14 → ~3–5 ms | tiny ranking changes; must re-validate | S | Maybe |
| 6 | Remove proxy hops | Pi → SSH tunnel → nezopt: direct LAN/WireGuard to :8098 | ~20 ms per scan, more for big uploads | exposure: the worker must stay private | S | Maybe |
| 7 | Hand-written io_uring server, multishot accept/recv | uvicorn front | none measurable: the model is 95 % of the time and requests are serialized by design | high | L | **Don't** |
| 8 | Conntrack | nezopt 1.8k / 262k, Pi 0.5k / 65k | none | — | — | Don't (Oracle micro: see the Pokoin doc) |

Bigger, non-video wins worth knowing:
- Clients should send 640–960 px images. Android does (640 px q85); the web desk sends 1600 px and double-encodes it.
- Run the GPU worker on :8099 when the GPU is free. The CPU path was chosen to keep the GPU free.

**Prototype:** BattleScan branch `scan-speed` (worktree `~/Projects/.worktrees/BattleScan/scan-speed`):
- `72329b8` puts the live `server/` into git as it runs.
- `aadfc98` adds `CARDSCAN_DETECTOR=onnx`, `CARDSCAN_TORCH_THREADS`, `CARDSCAN_JPEG_DRAFT=1`, all defaults unchanged.
- Benchmarks are `bench/httpbench.sh` (three instances, interleaved, curl) and `bench/scanprof2.py` (stages).

**Not merged or deployed.** To enable it after the owner's OK, add `CARDSCAN_DETECTOR=onnx` and `CARDSCAN_JPEG_DRAFT=1`
to `nezopt-cardscan.service`, running from the branch's code.

**Recommendation: do #1 + #3 now (≈ 3.5× faster on average, tail from seconds to < 0.4 s); #4–#6 later; skip io_uring.**

## 2. CardRails

**How it works:**
- **iPhone:** recognizes on device (Core ML + Milo + the Rust `pokoin_mobile_core`, ~59 ms per frame, search ~0.1 ms). It uploads only JPEG crops at commit.
- **Android:** local recognizer, falling back to `/api/scan/identify` (640 px q85).
- **Web desk:** 1600 px, double-encoded JPEG, strictly serial per photo.
- **`cardrails-api`:** Rust/axum/sqlx with jemalloc, in Docker on nezopt behind a Cloudflare tunnel, Postgres 17 (247 MB).
- **Price files:** PKP3, 12-byte sorted rows built nightly by `pokoin-scanner/native/price_book`, binary-searched on the client.
- **Catalogs:** binary packs served under `/v1/catalogs`.

The API itself answers in ~1 ms warm. Every public call pays ~70 ms of tunnel RTT.

| # | Finding | Gain | Verdict |
| --- | --- | --- | --- |
| 1 | The CardTrader and Pokoin syncs rewrite the same 12,251 rows every cycle (`cardtrader.rs:619`, `pokoin_sync.rs:1277`): 11.4M updates on a 15.6k-row table, 25.9 GB of Postgres writes in 2 days, and `version` bumps that cause edit conflicts | removes ~13 GB/day of writes and the conflicts | **Do** (a correctness fix more than a speed fix) |
| 2 | Cloudflare never caches the immutable catalogs or price files (`DYNAMIC`); the price file is gzipped on the fly per request (0.6 s vs 22 ms) | first-run downloads stop using nezopt's uplink; price fetch 0.6 s → ~0 | **Do**: cache rules plus precompressed (zstd/gzip) price files |
| 3 | Scan-adjacent bytes and round trips: web desk downscales to 960 px and encodes once, Android encodes once, crops are pre-uploaded while scanning, argon2 moves to `spawn_blocking` | 3–5× fewer upload bytes on the web; no stalls during login bursts | Do |
| 4 | Video's batched DB work | already the case for the price file (one file); #1 is the opposite problem | — |
| 5 | Release profile with LTO / `codegen-units=1` | a few % CPU | Maybe (cheap) |
| 6 | io_uring / multishot / hand-written server | none: the server is not the bottleneck | **Don't** |
| 7 | iOS cold paths: catalogs buffered whole in RAM with a 120 s resource timeout (134 MB FaB file), re-hashed on every load, price file not mmapped | first-run reliability on mobile | Maybe (measure on the iPhone first) |

## Not checked

- Real production scan latencies: almost no `/identify` traffic hit nezopt this week.
- iPhone on-device timings.
- The Pi/Cloudflare uplink throughput.
- Milo dynamic-batch export (#4) and the f16 catalog (#5): not prototyped.

The full CardRails study is `~/scratch/yt-test/cardrails-study.md` on nezopt.
