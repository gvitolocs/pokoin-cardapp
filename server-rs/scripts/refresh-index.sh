#!/bin/bash
# Index refresh with the placeholder guard, as run on 2026-10-30:
#   1. card-images audit    every placeholder face in the CDN objects tree
#   2. card-images recover  real pictures for them (dry run unless APPLY=1; then backups + derived files)
#   3. index-refresh plan   stage faces of cards the index lacks (placeholder faces are skipped)
#   4. embed_append.py      Milo embeddings of the staged faces (parity-checked against stored rows)
#   5. index-refresh enrich names, numbers, sets, catalogs
#   6. index-refresh purge  guard: the new index must have no placeholder rows (fails if it removes any)
# It never switches the live worker: point the nezopt-cardscan-rs drop-in at the printed index and restart.
#
# Usage: refresh-index.sh BASE_INDEX NEW_INDEX WORK_DIR
#   env: APPLY=1 to write recovered pictures; DAILY=<daily export dir> (default: latest)
set -euo pipefail
base=$1 new=$2 work=$3
here=$(cd "$(dirname "$0")/.." && pwd)
bin=$here/target/release
objects=${OBJECTS:-/home/nez/data/pokoin-leftovers/objects}
catalogs=${CATALOGS:-/home/nez/data/cardrails-catalogs/current}
daily=${DAILY:-$(ls -d /home/nez/data/pokoin-catalog-refresh/daily/*/ | tail -1)}
tcg=${TCG_DATASETS:-/home/nez/mnt/mybook/datasets/tcg}
python=${EMBED_PYTHON:-/home/nez/Projects/Startup/Pokoin/pokoin-scanner/.venv/bin/python}
capped=(systemd-run --user --scope -q -p CPUQuota=400%)

[[ -e $new ]] && { echo "$new exists" >&2; exit 1; }
mkdir -p "$work"
(cd "$here" && cargo build -q --release -p card-images -p index-refresh)

# pokoin_pokemon_expansions.official_id maps CardTrader expansions to pokemontcg/TCGdex set ids.
docker exec -i pokoin-marketplace-postgres-15t psql -U pokoin_marketplace -d pokoin_marketplace -At -F $'\t' -c \
  "select expansion_id, game_id, lower(official_id), lower(coalesce(code,'')), name, nationality
   from pokoin_pokemon_expansions where expansion_id is not null order by 1" > "$work/expansions.tsv"

"${capped[@]}" "$bin/card-images" audit --objects "$objects" --out "$work/audit.jsonl" --threads 4
apply=()
[[ ${APPLY:-0} == 1 ]] && apply=(--apply --backup "${BACKUP:-${objects%/objects}/backup-placeholders-$(date -u +%F)}")
"$bin/card-images" recover --audit "$work/audit.jsonl" --objects "$objects" --blueprints "$daily" \
  --expansions "$work/expansions.tsv" --tcgdex "$tcg/pokoin-cdn" --pokemontcg "$tcg/pokemon-tcg-data" \
  --scryfall "$tcg/scryfall-default-cards.jsonl.gz" --out "$work/recover" --download "${apply[@]}"

: > "$work/no-placeholder-list.txt"
"$bin/index-refresh" plan --index "$base" --catalogs "$catalogs" --blueprints "$daily" --objects "$objects" \
  --placeholders "$work/no-placeholder-list.txt" --out "$work/plan"
"${capped[@]}" "$python" "$here/scripts/embed_append.py" "$base" "$work/plan/faces" "$work/plan/allow.txt" "$new"
"$bin/index-refresh" enrich --index "$new" --cardmeta "$work/plan/cardmeta.jsonl"

"${capped[@]}" "$bin/index-refresh" purge --index "$new" --objects "$objects" --out "$work/guard" --threads 4
removed=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["rows_in"] - json.load(open(sys.argv[1]))["rows_out"])' \
  "$work/guard/purge-report.json")
rm -rf "$work/guard"
if [[ $removed != 0 ]]; then
  echo "guard: $new still has $removed placeholder rows; use index-refresh purge on it before going live" >&2
  exit 1
fi
echo "ready: $new"
