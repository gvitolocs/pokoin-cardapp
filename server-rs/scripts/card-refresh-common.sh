#!/bin/bash
# Shared runtime paths and thermal/CPU limits. Source from an entry point.
# shellcheck disable=SC2034 # Runtime globals consumed by sourcing entry points.
refresh_paths() {
  script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  if [[ -x $script_dir/card-images ]]; then
    bin=$script_dir
  else
    bin=$(cd "$script_dir/.." && pwd)/target/release
  fi
  objects=${OBJECTS:-/home/nez/data/pokoin-leftovers/objects}
  catalogs=${CATALOGS:-/home/nez/data/cardrails-catalogs/current}
  daily=${DAILY:-$(find /home/nez/data/pokoin-catalog-refresh/daily -mindepth 1 -maxdepth 1 -type d | sort | tail -1)}
  tcg=${TCG_DATASETS:-/home/nez/mnt/mybook/datasets/tcg}
  python=${EMBED_PYTHON:-/home/nez/Projects/Startup/Pokoin/pokoin-scanner/.venv/bin/python}
}
heavy() {
  local temp
  while :; do
    temp=$({ sensors 2>/dev/null || true; } | sed -nE 's/^Tctl: *\+([0-9]+).*/\1/p')
    [[ ${temp:-0} -lt 90 ]] && break
    echo "thermal: Tctl=$temp; waiting" >&2
    sleep 30
  done
  systemd-run --user --scope -q -p CPUQuota=400% "$@"
}
expansions() {
  if [[ -n ${EXPANSIONS:-} ]]; then
    cp "$EXPANSIONS" "$1"
  else
    docker exec -i pokoin-marketplace-postgres-15t psql -U pokoin_marketplace -d pokoin_marketplace -At -F $'\t' -c \
      "select expansion_id, game_id, lower(official_id), lower(coalesce(code,'')), name, nationality
       from pokoin_pokemon_expansions where expansion_id is not null order by 1" > "$1"
  fi
}
recover_images() {
  local audit=$1 work=$2
  shift 2
  heavy "$bin/card-images" recover --audit "$audit" --objects "$objects" --blueprints "$daily" \
    --expansions "$work/expansions.tsv" --tcgdex "$tcg/pokoin-cdn" --pokemontcg "$tcg/pokemon-tcg-data" \
    --scryfall "$tcg/scryfall-default-cards.jsonl.gz" --out "$work/recover" --download "$@"
}
