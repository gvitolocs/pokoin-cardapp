#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CLONES="$ROOT/clones"
NOTES="$ROOT/notes"
mkdir -p "$CLONES" "$NOTES"

repos=(
  "CollectorVision|https://github.com/HanClinto/CollectorVision.git"
  "Pokemon-Card-Scanning-Webapp|https://github.com/ShreyShingala/Pokemon-Card-Scanning-Webapp.git"
  "TCG|https://github.com/qtran1018/TCG.git"
)

fail=0
pids=()

clone_one() {
  local name="$1" url="$2"
  local dest="$CLONES/$name"
  local log="$NOTES/clone-$name.log"
  if [[ -d "$dest/.git" ]]; then
    echo "SKIP $name (already cloned)" | tee "$log"
    return 0
  fi
  echo "CLONE $name"
  if git clone --depth 1 --single-branch "$url" "$dest" >"$log" 2>&1; then
    echo "OK $name"
  else
    echo "FAIL $name — see $log" >&2
    return 1
  fi
}

for entry in "${repos[@]}"; do
  name="${entry%%|*}"
  url="${entry##*|}"
  clone_one "$name" "$url" &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    fail=1
  fi
done

exit "$fail"
