"""Append Milo embeddings of staged faces to a copy of cdn_milo.

Python only because the Milo embedder (CollectorVision NeuralEmbedder) is Python; same
preprocessing and row format as pokoin-scanner/training/cardapp/embed_cdn_milo.py
(--soften-sample off, --region-pool full). Checks parity on existing rows first.
"""
import argparse, json, random, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, "/home/nez/Projects/Startup/Pokoin/pokoin-scanner/clones/CollectorVision")
from collector_vision.embedders.neural import NeuralEmbedder

parser = argparse.ArgumentParser(description=__doc__)
for arg in ("old_dir", "faces", "allow", "out_dir"):
    parser.add_argument(arg, type=Path)
parser.add_argument("--reembed", type=Path)
parser.add_argument("--noimage", type=Path)
args = parser.parse_args()
old_dir, faces, allow, out_dir = args.old_dir, args.faces, args.allow, args.out_dir
gallery = [Path("/home/nez/Projects/Startup/Pokoin/PokoinTest/index/cdn_images"),
           Path("/home/nez/Projects/Startup/Pokoin/PokoinTest/index/cdn_images_delta")]
emb = NeuralEmbedder(checkpoint="/home/nez/Projects/Startup/Pokoin/pokoin-scanner/runtime/teacher/milo.onnx",
                     batch_size=8, num_threads=4, provider="cpu")
old = np.load(old_dir / "embeddings.npy").astype(np.float32)
lines = [l for l in (old_dir / "metadata.jsonl").read_text().splitlines() if l.strip()]
assert old.shape == (len(lines), 128), old.shape

def l2(a): return (a / np.clip(np.linalg.norm(a, axis=-1, keepdims=True), 1e-8, None)).astype(np.float32)

rng = random.Random(29); cos = []
for i in rng.sample(range(len(lines)), 400):
    if not np.any(old[i]): continue
    fn = json.loads(lines[i])["filename"]
    src = next((g / fn for g in gallery if (g / fn).is_file()), None)
    if src is None: continue
    v = l2(np.asarray(emb.embed([Image.open(src).convert("RGB")]), np.float32))[0]
    cos.append(float(v @ l2(old[i:i+1])[0]))
    if len(cos) >= 30: break
print(f"parity n={len(cos)} min={min(cos):.6f} mean={np.mean(cos):.6f}", flush=True)
assert len(cos) >= 20 and min(cos) > 0.999, "embedder does not reproduce stored vectors"

existing = {str(json.loads(line)["ct_id"]): i for i, line in enumerate(lines)}
reembed = sorted({l.strip() for l in args.reembed.read_text().splitlines() if l.strip()}, key=int) if args.reembed else []
for ct in reembed:
    assert ct in existing, f"reembed ct_id {ct} missing from index"
    assert not np.any(old[existing[ct]]), f"reembed ct_id {ct} has a non-zero vector"
ids = sorted({l.strip() for l in allow.read_text().splitlines() if l.strip()}, key=int)
by_ct = {p.name.split("_", 1)[0]: p for p in faces.iterdir()}
rows, vecs, t0 = [], [], time.time()
for s in range(0, len(ids), 8):
    batch = [by_ct[c] for c in ids[s:s+8]]
    imgs = [Image.open(p).convert("RGB") for p in batch]
    vecs.append(l2(np.asarray(emb.embed(imgs), np.float32).reshape(len(imgs), 128)))
    for p in batch:
        ct = p.name.split("_", 1)[0]
        slug = p.stem.split("_", 1)[1] if "_" in p.stem else p.stem
        rows.append({"ct_id": ct, "public_id": str(int(ct) * 2), "filename": p.name, "name": slug.replace("-", " ")})
replaced = 0
for s in range(0, len(reembed), 8):
    batch = [by_ct[c] for c in reembed[s:s+8]]
    imgs = [Image.open(p).convert("RGB") for p in batch]
    replacements = l2(np.asarray(emb.embed(imgs), np.float32).reshape(len(imgs), 128))
    for ct, p, vector in zip(reembed[s:s+8], batch, replacements):
        i = existing[ct]
        old[i] = vector
        row = json.loads(lines[i])
        row["image"] = p.name
        lines[i] = json.dumps(row)
        replaced += 1
noimage_rows = []
seen = set(existing) | {r["ct_id"] for r in rows}
if args.noimage:
    for line in args.noimage.read_text().splitlines():
        if not line.strip(): continue
        row = json.loads(line)
        ct = str(row["ct_id"])
        if ct in seen: continue
        row["image"] = "none"
        noimage_rows.append(row)
        seen.add(ct)
new = np.concatenate(vecs) if vecs else np.zeros((0, 128), np.float32)
out_dir.mkdir(parents=True, exist_ok=False)
arr = np.concatenate([old, new, np.zeros((len(noimage_rows), 128), np.float32)])
np.save(out_dir / "embeddings.npy", arr)
with (out_dir / "metadata.jsonl").open("w") as fh:
    for l in lines: fh.write(l + "\n")
    for r in rows + noimage_rows: fh.write(json.dumps(r) + "\n")
(out_dir / "progress.json").write_text(json.dumps({"mode": "append", "provider": "cpu", "embedded": int(arr.shape[0]),
    "appended": len(rows), "replaced": replaced, "noimage_appended": len(noimage_rows), "base": str(old_dir), "parity_min_cos": round(min(cos), 6), "elapsed_s": round(time.time() - t0, 1)}, indent=2))
print(f"appended={len(rows)} replaced={replaced} noimage_appended={len(noimage_rows)} total={arr.shape[0]} elapsed={time.time()-t0:.1f}s", flush=True)
