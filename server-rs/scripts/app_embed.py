#!/usr/bin/env python3
"""Parity-checked CPU MiloCNN updates for the positional app catalogs."""
import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
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


def worst_rows(cosines, ids, images, count=3):
    """Up to `count` lowest-parity rows as (cosine, row id, image key), worst first; ties by row id."""
    ranked = sorted(((float(cosine), row_id, image) for cosine, row_id, image in zip(cosines, ids, images)),
                    key=lambda row: (row[0], row[1]))
    return ranked[:count]


def image_key(image):
    """The objects-relative file of a catalog image URL, or '' when it names no file we can read."""
    key = image[len(PREFIX):] if image and image.startswith(PREFIX) else ''
    if not key or Path(key).is_absolute() or '..' in Path(key).parts:
        return ''
    return key


def _parse_stamp(value):
    """Epoch seconds of an ISO 8601 stamp text; None when the value is missing or unparsable."""
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value[:-1] + '+00:00' if value.endswith('Z') else value).timestamp()
    except ValueError:
        return None


def catalog_stamp(catalogs_dir, index):
    """When the catalog's own vectors were computed: (epoch seconds, source) or (None, 'no stamp')."""
    sidecar = None
    try:
        sidecar = json.loads((catalogs_dir / 'embedded-at.json').read_text())
    except (OSError, ValueError):
        pass
    run_start = sidecar.get('runStart') if isinstance(sidecar, dict) else None
    for value, source in ((run_start, 'embedded-at.json'), (index.get('generatedAt'), 'index.generatedAt')):
        parsed = _parse_stamp(value)
        if parsed is not None:
            return parsed, source
    return None, 'no stamp'


def comparable_rows(rows, vectors, noimage, audit, image_key, mtime_of, stamp=None):
    """Rows whose stored vector and picture can be compared, and among them those whose picture is newer."""
    comparable, changed = [], []
    for i, row in enumerate(rows):
        if row['id'] in noimage or not np.any(vectors[i]):
            continue
        key = image_key(row.get('image'))
        if not key or key in audit:
            continue
        mtime = mtime_of(key)
        if mtime is None:
            continue
        comparable.append(i)
        if stamp is not None and mtime > stamp:
            changed.append(i)
    return comparable, changed


def refresh_indices(rows, vectors, noimage, audit, image_key, mtime_of, stamp):
    """Row indices whose picture is newer than the catalog's stamp, so its vector describes a retired file."""
    return comparable_rows(rows, vectors, noimage, audit, image_key, mtime_of, stamp)[1]


def parity_step(rows, vectors, noimage, audit, image_key, mtime_of, stamp, embed_rows, normalize,
                verify=30, seed=29, threshold=0.995, max_skipped_fraction=0.20, stamp_text='none'):
    """Compare `verify` sampled rows whose picture has not changed with a fresh embedding of the same file.

    Returns (code, lines, pick, skipped); `lines` are (stream, text) pairs for the caller to print.
    """
    comparable_all, skipped = comparable_rows(rows, vectors, noimage, audit, image_key, mtime_of, stamp)
    if len(comparable_all) < verify:
        return 2, [('stderr', 'not enough parity rows')], [], skipped
    changed = set(skipped)
    eligible = [i for i in comparable_all if i not in changed]
    if len(eligible) < verify or len(skipped) > max_skipped_fraction * len(rows):
        refusal = f'parity: not enough unchanged rows: {len(eligible)} eligible, {len(skipped)} skipped'
        return 3, [('stderr', refusal)], [], skipped
    pick = random.Random(seed).sample(eligible, verify)
    fresh = embed_rows(pick)
    cosine = np.sum(fresh * normalize(vectors[pick].astype(np.float32)), axis=1)
    minimum = float(cosine.min())
    lines = [('stdout', f'parity n={len(pick)} min={minimum:.6f}'),
             ('stdout', f'parity below 0.995: {sum(1 for v in cosine if float(v) < 0.995)} of {len(pick)}')]
    for c, row_id, key in worst_rows(cosine, [rows[i]['id'] for i in pick],
                                     [image_key(rows[i]['image']) for i in pick]):
        lines.append(('stdout', f'parity row id={row_id} cos={c:.6f} image={key}'))
    skipped_line = f'parity skipped {len(skipped)} changed rows'
    if skipped:
        skipped_line += ': ' + ' '.join(str(rows[i]['id']) for i in skipped[:10])
    lines.append(('stdout', skipped_line))
    rule = ('a row is skipped only when its image file is newer than the stamp, an image replaced with its old '
            'modification time kept is still compared and may still fail')
    lines.append(('stdout', f'parity stamp {stamp_text}; {rule}'))
    return (2 if not np.isfinite(minimum) or minimum < threshold else 0), lines, pick, skipped


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
    stamp, stamp_source = catalog_stamp(args.catalogs, index)
    stamp_text = ('no stamp' if stamp is None else
                  f'{datetime.fromtimestamp(stamp, timezone.utc):%Y-%m-%dT%H:%M:%SZ} from {stamp_source}')

    def mtime_of(key):
        path = args.objects / key
        return path.stat().st_mtime if path.is_file() else None

    cat, rows, vectors = catalogs['pokemon_western']
    code, lines, _, _ = parity_step(
        rows, vectors, set(cat.get('noImagePublicIDs', [])), audit, image_key, mtime_of, stamp,
        lambda indices: embed([args.objects / image_key(rows[i]['image']) for i in indices]),
        card_cnn.l2_normalize, args.verify, stamp_text=stamp_text)
    for stream, text in lines:
        print(text, file=sys.stderr if stream == 'stderr' else sys.stdout, flush=True)
    if code:
        return code
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
            replaced = appended = missing = refreshed = 0
            for i, row in enumerate(rows):
                if row['id'] not in noimage and np.any(vectors[i]):
                    continue
                k = image_key(row.get('image'))
                # Rows appended without a face have an empty image until a later daily export.
                if not k and row['id'].isdigit() and int(row['id']) % 2 == 0:
                    k = next(iter(faces.get(int(row['id']) // 2, [])), '')
                if k and k not in audit and (args.objects / k).is_file():
                    pending.append({'catalog': cid, 'id': row['id'], 'image': PREFIX + k})
                    paths.append(args.objects / k)
                    replaced += 1
            # Pictures replaced since the catalog's own stamp: its vector no longer describes the file the app
            # serves, so re-embed them into this revision. Parity skips exactly these rows for the same reason.
            queued = {record['id'] for record in pending}
            for i in refresh_indices(rows, vectors, noimage, audit, image_key, mtime_of, stamp):
                if rows[i]['id'] in queued:
                    continue
                k = image_key(rows[i].get('image'))
                pending.append({'catalog': cid, 'id': rows[i]['id'], 'image': PREFIX + k})
                paths.append(args.objects / k)
                queued.add(rows[i]['id'])
                refreshed += 1
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
                preferred = image_key(preferred) if preferred.startswith(PREFIX) else preferred
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
            counts[cid] = {'replaced': replaced, 'appended': appended, 'no_image_appended': missing,
                           'refreshed': refreshed}
    print(json.dumps(counts), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
