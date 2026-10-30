#!/bin/bash
# Entry point for pokoin-all-cardtrader-daily. DRY_RUN never changes objects or the worker.
set -euo pipefail
# shellcheck source=scripts/card-refresh-common.sh
source "$(dirname "$0")/card-refresh-common.sh"
refresh_paths
state=${CARD_STATE:-/home/nez/data/card-images}
dry=${DRY_RUN:-0}
date_id=$(date -u +%Y%m%d)
stamp=$(date -u +%Y-%m-%dT%H%M%SZ)
mkdir -p "$state"
exec 9> "$state/lock"
flock -n 9 || { echo 'card refresh already running' >&2; exit 1; }
run=$state/runs/$stamp
mkdir -p "$run"
exec > >(tee -a "$run/run.log") 2>&1
unique_path() {
  local candidate=$1 suffix=2
  while [[ -e $candidate || -L $candidate ]]; do candidate=$1-$suffix; ((suffix+=1)); done
  printf '%s\n' "$candidate"
}
previous=()
if [[ -f $state/audit.jsonl ]]; then
  cp -p "$state/audit.jsonl" "$state/audit.prev.jsonl"
  previous=(--previous "$state/audit.prev.jsonl")
fi
heavy "$bin/card-images" audit --objects "$objects" --out "$run/audit.before.jsonl" --threads 4 "${previous[@]}"
expansions "$run/expansions.tsv"
apply=()
[[ $dry == 0 ]] && apply=(--apply --backup "/home/nez/data/pokoin-leftovers/backup-placeholders-$stamp")
recover_images "$run/audit.before.jsonl" "$run" "${apply[@]}"
heavy "$bin/card-images" audit --objects "$objects" --previous "$run/audit.before.jsonl" --out "$run/audit.jsonl" --threads 4
cp -p "$run/audit.jsonl" "$state/audit.next.jsonl"
mv "$state/audit.next.jsonl" "$state/audit.jsonl"
# Resolve live index from the service, or read-only health in dry mode (no systemctl).
index_root=${INDEX_ROOT:-/home/nez/Projects/Startup/Pokoin/PokoinTest/index}
base=${BASE_INDEX:-}
if [[ -z $base ]]; then
  if [[ $dry == 1 ]]; then
    live_name=$(curl -fsS --max-time 10 http://127.0.0.1:8098/health | jq -er .index)
    [[ $live_name != */* ]] || { echo 'invalid health index basename' >&2; exit 1; }
    base=$index_root/$live_name
  else
    environment=$(systemctl --user show nezopt-cardscan-rs -p Environment --value)
    base=$(printf '%s\n' "$environment" | tr ' ' '\n' | sed -n 's/^CDN_MILO=//p')
  fi
fi
[[ -d $base && -n $base ]] || { echo "missing live index: $base" >&2; exit 1; }
if [[ $dry == 1 ]]; then
  mkdir -p "$state/dry"
  new=$(unique_path "$state/dry/cdn_milo_$date_id")
else
  new=$(unique_path "$index_root/cdn_milo_$date_id")
fi
SKIP_AUDIT=1 AUDIT=$state/audit.jsonl "$script_dir/refresh-index.sh" "$base" "$new" "$run/index"
swapped=false
rollback_needed=0
dropins=$HOME/.config/systemd/user/nezopt-cardscan-rs.service.d
rollback() {
  local status=$?
  if [[ $rollback_needed == 1 ]]; then
    echo 'swap failed; restoring previous drop-ins' >&2
    rm -f "$dropins/60-index.conf" "$dropins/.60-index.conf.tmp" "$dropins"/60-index-*.conf
    cp -a "$run/dropins/." "$dropins/"
    systemctl --user daemon-reload
    systemctl --user restart nezopt-cardscan-rs
    exit 1
  fi
  return "$status"
}
trap rollback EXIT
trap 'exit 1' INT TERM HUP
if [[ -d $new && $dry == 0 && ${SWAP:-1} == 1 ]]; then
  mkdir -p "$run/dropins" "$dropins"
  shopt -s nullglob
  old_dropins=("$dropins/60-index.conf" "$dropins"/60-index-*.conf)
  for file in "${old_dropins[@]}"; do [[ ! -e $file ]] || cp -a "$file" "$run/dropins/"; done
  rollback_needed=1
  printf '# Previous CDN_MILO=%s\n[Service]\nEnvironment=CDN_MILO=%s\n' "$base" "$new" > "$dropins/.60-index.conf.tmp"
  mv "$dropins/.60-index.conf.tmp" "$dropins/60-index.conf"
  rm -f "$dropins"/60-index-*.conf
  systemctl --user daemon-reload
  systemctl --user restart nezopt-cardscan-rs
  expected=$(wc -l < "$new/metadata.jsonl")
  healthy=0
  for ((attempt=0; attempt<90; attempt++)); do
    if curl -fsS --max-time 5 http://127.0.0.1:8098/health > "$run/health.json" && \
       jq -e --arg index "$(basename "$new")" --argjson n "$expected" '.index==$index and .n==$n' "$run/health.json" >/dev/null; then
      healthy=1; break
    fi
    sleep 2
  done
  [[ $healthy == 1 ]]
  # web.rs exposes /identify directly; /api/scan is the public proxy prefix.
  curl -fsS --max-time 120 -F "file=@$objects/370861_numel.jpg" \
    'http://127.0.0.1:8098/identify?top_k=1' > "$run/identify.json"
  jq -e '(.hits[0].ct_id|tostring)=="370861"' "$run/identify.json" >/dev/null
  rollback_needed=0
  swapped=true
fi
if [[ -d $new && $dry == 0 ]]; then
  printf '%s\n' "$new" >> "$state/indexes.txt"
  # Retention only removes tracked directories, never the current or previous worker index.
  mapfile -t tracked < "$state/indexes.txt"
  : > "$state/indexes.next"
  for ((i=0; i<${#tracked[@]}; i++)); do
    candidate=${tracked[i]}
    if (( i < ${#tracked[@]}-5 )) && [[ $candidate != "$base" && $candidate != "$new" && $candidate == "$index_root"/cdn_milo_* && -d $candidate && ! -L $candidate ]]; then
      rm -rf -- "$candidate"
    else
      printf '%s\n' "$candidate" >> "$state/indexes.next"
    fi
  done
  mv "$state/indexes.next" "$state/indexes.txt"
fi
heavy "$python" "$script_dir/app_embed.py" --catalogs "$catalogs" --objects "$objects" \
  --audit "$state/audit.jsonl" --cardmeta "$run/index/plan/cardmeta.jsonl" --out "$run/app-embeddings.jsonl"
if [[ $dry == 1 ]]; then catalog_out=$(unique_path "$state/dry/$date_id-auto-cards");
else catalog_out=$(unique_path "/home/nez/data/cardrails-catalogs/$date_id-auto-cards"); fi
heavy "$bin/card-images" catalog-mark --catalogs "$catalogs" --audit "$state/audit.jsonl" --out "$catalog_out" \
  --only pokemon_western,pokemon_japanese,pokemon_chinese --embeddings "$run/app-embeddings.jsonl"
staged='' published=false
# Ignore provenance-only changes; retain a revision only if rows/vectors/no-image state changed.
if jq -e -s '.[0].catalogs != .[1].catalogs' "$catalogs/index.json" "$catalog_out/index.json" >/dev/null; then
  staged=$catalog_out
  if [[ $dry == 0 && ${CARDRAILS_CATALOG_AUTOPUBLISH:-0} == 1 ]]; then
    current=/home/nez/data/cardrails-catalogs/current
    ln -s "$catalog_out" "$current.next-$stamp"
    mv -Tf "$current.next-$stamp" "$current"
    published=true
  fi
  echo "catalog staged: $staged published=$published"
else
  rm -rf -- "$catalog_out"
fi
# Unpublished nightly revisions pile up while publishing waits for the owner: keep the newest 3, never the live one.
if [[ $dry == 0 ]]; then
  live_catalogs=$(readlink -f /home/nez/data/cardrails-catalogs/current)
  mapfile -t autos < <(find /home/nez/data/cardrails-catalogs -mindepth 1 -maxdepth 1 -type d -name '*-auto-cards*' | sort)
  for ((i=0; i<${#autos[@]}-3; i++)); do
    [[ ${autos[i]} == "$live_catalogs" ]] || rm -rf -- "${autos[i]}"
  done
fi
active=$base
[[ ! -d $new ]] || active=$new
rows=$(wc -l < "$active/metadata.jsonl")
no_image=$(jq -s '[.[]|select(.image=="none")]|length' "$active/metadata.jsonl")
jq -n --slurpfile before "$run/audit.before.jsonl.report.json" --slurpfile audit "$run/audit.jsonl.report.json" --slurpfile recovery "$run/recover/report.json" \
  --argjson rows "$rows" --argjson no_image "$no_image" --argjson swapped "$swapped" \
  --arg staged "$staged" --argjson published "$published" \
  '{audit:{examined:$before[0].examined,carried_over:$before[0].carried_over,placeholders:([$audit[0].games[].placeholders]|add),recheck_examined:$audit[0].examined,recheck_carried_over:$audit[0].carried_over}, recovered:($recovery[0].games|map_values(.recovered)), missing:($recovery[0].games|map_values(.missing)), index:{rows:$rows,no_image:$no_image}, swapped:$swapped, catalog:{staged:$staged,published:$published}}' > "$run/summary.json"
printf 'card-refresh summary: %s\n' "$(jq -c . "$run/summary.json")"
