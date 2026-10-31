#!/usr/bin/env python3
"""Parity-checked CPU MiloCNN updates for the positional app catalogs."""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import onnxruntime as ort

SCANNER = Path('/home/nez/Projects/Startup/Pokoin/pokoin-scanner')
sys.path.insert(0, str(SCANNER / 'training/cardapp'))
import card_cnn

PREFIX = 'https://cdn.pokoin.com/'
TARGETS = ('pokemon_western', 'pokemon_japanese', 'pokemon_chinese')


def pins(value):
    if isinstance(value, dict):
        if isinstance(value.get('catalog_sha256'), str):
            yield value['catalog_sha256']
        for v in value.values():
            yield from pins(v)
    elif isinstance(value, list):
        for v in value:
            yield from pins(v)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('catalogs', 'objects', 'audit', 'cardmeta', 'out'):
        parser.add_argument('--' + flag, type=Path, required=True)
    parser.add_argument('--verify', type=int, default=30)
    args = parser.parse_args()
    if args.verify < 1:
        parser.error('--verify must be positive')
    index = json.loads((args.catalogs / 'index.json').read_text())
    model = SCANNER / 'runtime/fast-models-cnn/milo.onnx'
    if (hashlib.sha256(model.read_bytes()).hexdigest() != index['model']['onnxSha256']
            or index['model']['size'] != 448 or index['model']['dim'] != 128):
        raise SystemExit('catalog model mismatch')
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model), options, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    def embed(paths):
        vectors = []
        for start in range(0, len(paths), 8):
            tensors = []
            for path in paths[start:start + 8]:
                with Image.open(path) as image:
                    tensors.append(card_cnn.preprocess_pil(image))
            vectors.extend(card_cnn.l2_normalize(np.asarray(
                session.run(None, {input_name: np.stack(tensors)})[0], dtype=np.float32)))
        return np.asarray(vectors, dtype=np.float32)

    def key(image):
        key = image[len(PREFIX):] if image and image.startswith(PREFIX) else ''
        if not key or Path(key).is_absolute() or '..' in Path(key).parts:
            return ''
        return key

    audit = {json.loads(line)['key'] for line in args.audit.read_text().splitlines() if line.strip()}
    pinned = set()
    for manifest in (args.catalogs / 'packs').rglob('*manifest*.json'):
        pinned.update(pins(json.loads(manifest.read_text())))
    catalogs = {}
    for cat in index['catalogs']:
        if cat['id'] not in TARGETS:
            continue
        cards = json.loads((args.catalogs / cat['cards']['path']).read_text())
        rows = [dict(zip(cards['fields'], row)) for row in cards['rows']]
        vectors = np.fromfile(args.catalogs / cat['embeddings']['path'], dtype='<f2').reshape(len(rows), 128)
        catalogs[cat['id']] = (cat, rows, vectors)
    cat, rows, vectors = catalogs['pokemon_western']
    noimage = set(cat.get('noImagePublicIDs', []))
    eligible = [i for i, row in enumerate(rows)
                if row['id'] not in noimage and np.any(vectors[i])
                and key(row.get('image')) not in audit and key(row.get('image'))
                and (args.objects / key(row.get('image'))).is_file()]
    if len(eligible) < args.verify:
        print('not enough parity rows', file=sys.stderr)
        return 2
    pick = random.Random(29).sample(eligible, args.verify)
    fresh = embed([args.objects / key(rows[i]['image']) for i in pick])
    cosine = np.sum(fresh * card_cnn.l2_normalize(vectors[pick].astype(np.float32)), axis=1)
    minimum = float(cosine.min())
    print(f'parity n={len(pick)} min={minimum:.6f}', flush=True)
    if not np.isfinite(minimum) or minimum < 0.995:
        return 2
    meta = [json.loads(line) for line in args.cardmeta.read_text().splitlines() if line.strip()]
    faces = {}
    for path in sorted(args.objects.iterdir()):
        if path.is_file() and path.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp') and '_homepage' not in path.stem:
            ct = path.name.split('_', 1)[0]
            if ct.isdigit():
                faces.setdefault(int(ct), []).append(path.name)
    for candidates in faces.values():
        candidates.sort(key=lambda name: (('.jpg', '.jpeg', '.png', '.webp').index(Path(name).suffix.lower()), name))
    counts = {}
    with args.out.open('x') as out:
        for cid, (cat, rows, vectors) in catalogs.items():
            if cat['cards']['sha256'] in pinned:
                counts[cid] = {'pinned': True}
                continue
            noimage = set(cat.get('noImagePublicIDs', []))
            existing = {row['id'] for row in rows}
            pending, paths = [], []
            replaced = appended = missing = 0
            for i, row in enumerate(rows):
                if row['id'] not in noimage and np.any(vectors[i]):
                    continue
                k = key(row.get('image'))
                # Rows appended without a face have an empty image until a later daily export.
                if not k and row['id'].isdigit() and int(row['id']) % 2 == 0:
                    k = next(iter(faces.get(int(row['id']) // 2, [])), '')
                if k and k not in audit and (args.objects / k).is_file():
                    pending.append({'catalog': cid, 'id': row['id'], 'image': PREFIX + k})
                    paths.append(args.objects / k)
                    replaced += 1
            # Only cards CardTrader added after the catalog's newest card are appended (blueprint ids grow over
            # time). Older cards missing from a legacy catalog were left out by its membership rules (World
            # Championship Decks, starter decks, ...); adding them is a membership change, not a nightly one.
            newest = max((int(i) // 2 for i in existing if i.isdigit()), default=0)
            for card in meta:
                public = str(2 * int(card['ct_id']))
                if cid not in card.get('catalogs', []) or public in existing or int(card['ct_id']) <= newest:
                    continue
                existing.add(public)
                candidates = faces.get(int(card['ct_id']), [])
                preferred = card.get('image') or ''
                preferred = key(preferred) if preferred.startswith(PREFIX) else preferred
                k = preferred if preferred in candidates else next(iter(candidates), '')
                available = bool(k and k not in audit)
                record = {'catalog': cid, 'id': public,
                          'append': {'id': public, 'name': card.get('name'), 'number': card.get('n'),
                                     'set': card.get('set'), 'image': PREFIX + k if available else ''}}
                appended += 1
                if available:
                    pending.append(record)
                    paths.append(args.objects / k)
                else:
                    record['vector'] = None
                    out.write(json.dumps(record) + '\n')
                    missing += 1
            for start in range(0, len(paths), 8):
                for record, vector in zip(pending[start:start + 8], embed(paths[start:start + 8])):
                    record['vector'] = vector.tolist()
                    out.write(json.dumps(record) + '\n')
            counts[cid] = {'replaced': replaced, 'appended': appended, 'no_image_appended': missing}
    print(json.dumps(counts), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
