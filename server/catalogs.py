"""Read-only language/game galleries; requests never mutate another user's selection."""
from collections import OrderedDict
from pathlib import Path
import hashlib
import json
import numpy as np

class CatalogStore:
    def __init__(self, root: Path, model: Path):
        self.root = root
        self.manifest = json.loads((root / 'manifest.json').read_text())
        if hashlib.sha256(model.read_bytes()).hexdigest() != self.manifest['model_sha256']:
            raise RuntimeError('Catalogs require their original Milo model')
        self.entries = {e['id']: e for e in self.manifest['catalogs']}
        self.cache = OrderedDict()
        self.cache_limit = 8
        for key, entry in self.entries.items():
            if key != f"{entry['game']}_{entry['language']}" or '/' in key:
                raise RuntimeError('Invalid catalog manifest')
            for name, digest in entry['sha256'].items():
                if name not in ('embeddings.npy', 'metadata.jsonl'):
                    raise RuntimeError('Invalid catalog file')
                if hashlib.sha256((root / key / name).read_bytes()).hexdigest() != digest:
                    raise RuntimeError(f'Catalog integrity check failed: {key}/{name}')

    def public_entries(self):
        return [{k: e[k] for k in ('id','game','language','count','identity','detector')} for e in self.entries.values()]

    def get(self, key):
        if key not in self.entries:
            raise KeyError(key)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        path = self.root / key
        vectors = np.load(path / 'embeddings.npy', mmap_mode='r', allow_pickle=False)
        with (path / 'metadata.jsonl').open() as fh:
            cards = [json.loads(line) for line in fh if line.strip()]
        if vectors.shape != (len(cards),128) or len(cards) != self.entries[key]['count']:
            raise RuntimeError(f'Catalog dimensions mismatch: {key}')
        value = (vectors, cards)
        self.cache[key] = value
        while len(self.cache) > self.cache_limit:
            self.cache.popitem(last=False)
        return value
