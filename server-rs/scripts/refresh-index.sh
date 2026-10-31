#!/bin/bash
# Usage: refresh-index.sh BASE NEW WORK; SKIP_AUDIT=1 AUDIT=... for the daily driver.
set -euo pipefail
base=$1 new=$2 work=$3
# shellcheck source=scripts/card-refresh-common.sh
source "$(dirname "$0")/card-refresh-common.sh"
refresh_paths
[[ ! -e $new ]] || { echo "$new exists" >&2; exit 1; }
mkdir -p "$work"
if [[ ${MARK_BASE:-0} == 1 ]]; then
  heavy "$bin/index-refresh" mark --index "$base" --objects "$objects" --out "$work/base-marked" --threads 4
  base=$work/base-marked
fi
if [[ ${SKIP_AUDIT:-0} != 1 ]]; then
  expansions "$work/expansions.tsv"
  heavy "$bin/card-images" audit --objects "$objects" --out "$work/audit.jsonl" --threads 4
  apply=()
  [[ ${APPLY:-0} == 1 && ${DRY_RUN:-0} != 1 ]] && apply=(--apply --backup "${BACKUP:-${objects%/objects}/backup-placeholders-$(date -u +%F)}")
  recover_images "$work/audit.jsonl" "$work" "${apply[@]}"
  heavy "$bin/card-images" audit --objects "$objects" --previous "$work/audit.jsonl" --out "$work/audit.after.jsonl" --threads 4
fi
audit=${AUDIT:-$work/audit.after.jsonl}
# plan's detector guards all staged faces; audited ct_ids also exclude known placeholders.
jq -r '.ct_id // empty' "$audit" > "$work/placeholders.txt"
heavy "$bin/index-refresh" plan --index "$base" --catalogs "$catalogs" --blueprints "$daily" --objects "$objects" \
  --placeholders "$work/placeholders.txt" --out "$work/plan"
if [[ ! -s $work/plan/allow.txt && ! -s $work/plan/reembed.txt && ! -s $work/plan/noimage.jsonl ]]; then
  echo "unchanged: $base"
  exit 0
fi
heavy "$python" "$script_dir/embed_append.py" "$base" "$work/plan/faces" "$work/plan/allow.txt" "$new" \
  --reembed "$work/plan/reembed.txt" --noimage "$work/plan/noimage.jsonl"
"$bin/index-refresh" enrich --index "$new" --cardmeta "$work/plan/cardmeta.jsonl"
heavy "$bin/index-refresh" mark --index "$new" --objects "$objects" --out "$work/guard" --threads 4
if [[ $(jq .newly_marked "$work/guard/mark-report.json") != 0 ]]; then
  echo "guard failed: inspect $work/guard/mark-report.json" >&2
  exit 1
fi
echo "ready: $new"
