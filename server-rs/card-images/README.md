# card-images

Audit CDN face images and recover verified replacements. The shared
`card-placeholder` crate owns the existing detector, reference images and samples;
`index-refresh` uses the same detector.

```sh
cargo run --release -p card-images -- audit \
  --objects /path/to/copied-objects --out /path/to/audit.jsonl --threads 4

cargo run --release -p card-images -- recover \
  --audit /path/to/audit.jsonl --objects /path/to/copied-objects \
  --blueprints /path/to/daily-export --expansions /path/to/expansions.tsv \
  --tcgdex /path/to/pokoin-cdn --pokemontcg /path/to/pokemon-tcg-data \
  --scryfall /path/to/scryfall-default-cards.jsonl.gz --out /path/to/recovery
```

`audit` skips derivative and non-card directories at every depth. It reads image
headers, hashes faces and classifies Pokémon faces on worker threads. For other
games it classifies one representative of each identical-byte group spanning at
least three blueprint ids. The JSONL contains placeholders only. The adjacent
`audit.jsonl.report.json` has per-game file and placeholder counts, reasons, and
up to 50 unrecognised shared-byte groups. Undecodable files count as files but
are not labelled placeholders. Paths and reports are sorted deterministically.
Symlinks are not followed during traversal.

Use `--previous FILE` to carry unchanged placeholder rows and examine only files newer than FILE's mtime.
Deleted rows disappear and recovered rows are reclassified. Keep the mtime when copying snapshots (`cp -p`).
Reports add `examined` and `carried_over`; skipped old real files are absent from per-game counts. New byte-identical
groups can only be discovered among examined files plus carried placeholder rows. Run a full audit to include
older real files again.

`catalog-mark --catalogs DIR --audit FILE --out DIR [--only id,id] [--cos 0.995] [--embeddings FILE]`
exports a hard-linked revision with no-image rows preserved, zero f16 vectors and sorted `noImagePublicIDs`.
It accepts app embedding replacements/appends, keeps positional string/null fields, and skips catalogs pinned
by pack manifests. `catalog-purge` remains a legacy command. See the server README for the daily driver and
app embedding contract.

`recover` is offline and dry-run by default. It tries same-blueprint siblings,
CardTrader, TCGdex, pokemontcg.io and Scryfall in that order, subject to game and
stamp restrictions. Sibling directories are indexed once. Only relevant exports,
sets and card ids are retained in memory. The Scryfall mirror is discovered under
`--tcgdex/scryfall/<language>/<set>/<uuid>-0.jpg`; the back of a double-faced card
is not selected. Set names come from `_expansions-all.json`, with TSV names as a
fallback. All candidate images must decode, pass the detector and satisfy the
minimum size and aspect ratio rules. Reference datasets also require matching
names and collector numbers. Number matching uses the part before `/` and
preserves alphanumeric prefixes. Name folding removes a final parenthesised
suffix, lowercases, folds Latin accents and retains ASCII letters/digits.

Add `--download` to enable HTTP sources. Requests use rustls, a 20-second timeout,
two retries and a descriptive User-Agent. Requests are sequential (at most one
in flight). There is no cookie support or environment proxy discovery; URL
credentials are rejected on initial URLs and redirects. Decodable downloads are
cached by URL SHA-256 in `--out/cache/`.

Output files are `recovered.jsonl`, `missing.jsonl` and `report.json`. Each command
also prints its JSON summary. Recovery counts include `skipped` for audit keys
that are currently no longer placeholders. Missing-reason precedence is
`no_blueprint`, then `stamped_variant` after trying permitted sources, then
`mismatch`, `download_failed`, `download_disabled`, `no_source`. A failed source
does not prevent trying the next permitted source. Invalid image content alone
is `no_source`; a matching remote candidate with downloads disabled records
`download_disabled`.

Add `--apply --backup /path/to/backup` to replace faces and existing homepage and
preview images. Output and backup directories must be separate from the objects
tree. Backups use exclusive creation and existing backups are preserved. New
images are encoded to the destination extension (JPEG quality 92, PNG, lossless
WebP), then written through a temporary file and rename with original permissions.
Derived widths are preserved and heights follow the replacement's aspect ratio.
Derivatives finish before the face is replaced, so a failed derivative can be
retried while the face remains a placeholder. Each file replacement is atomic;
replacement of the face and its derivatives is not a multi-file transaction.
`sha256_new` describes the encoded face that apply writes, including in dry-run.
A repeated apply skips faces that are no longer placeholders.

```sh
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

Tests use local fixtures and make no HTTP requests.
