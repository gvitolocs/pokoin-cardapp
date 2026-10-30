#!/bin/bash
# End-to-end /identify latency of three builds of the scanner, interleaved (client side, curl).
set -u
B=/home/nez/Projects/Startup/Pokoin/BattleScan
W=/home/nez/Projects/.worktrees/BattleScan/scan-speed
export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES=-1 CARDSCAN_THREADS=4 \
  CDN_MILO=/home/nez/Projects/Startup/Pokoin/PokoinTest/index/cdn_milo \
  MILO_ONNX=$B/clones/CollectorVision/collector_vision/weights/milo.onnx \
  YOLO_PT=$B/clones/TCG/backend/models/card_detector.pt \
  YOLO_ONNX=$B/clones/TCG/backend/models/card_detector.onnx CARDSCAN_WEB=$B/web
pids=()
start() { # port, env...
  local port=$1; shift
  env "$@" $B/.venv/bin/uvicorn cdn_app:app --app-dir $W/server --host 127.0.0.1 --port $port --workers 1 >/tmp/scanbench-$port.log 2>&1 &
  pids+=($!)
}
trap 'kill ${pids[@]} 2>/dev/null' EXIT
start 18197 CARDSCAN_DETECTOR=pt
start 18198 CARDSCAN_DETECTOR=pt CARDSCAN_TORCH_THREADS=4 CARDSCAN_JPEG_DRAFT=1
start 18199 CARDSCAN_DETECTOR=onnx CARDSCAN_JPEG_DRAFT=1
for p in 18197 18198 18199; do for i in $(seq 1 90); do curl -fs -m 2 http://127.0.0.1:$p/health >/dev/null && break; sleep 1; done; done
imgs=$(ls $B/images/vinted/*.jpg | head -24; ls $B/images/*.png | head -6)
for p in 18197 18198 18199; do curl -s -o /dev/null -F file=@$B/images/vinted/04.jpg http://127.0.0.1:$p/identify; done  # warm-up
: > /tmp/scanbench.tsv
for rep in 1 2 3; do for f in $imgs; do for p in 18197 18198 18199; do
  out=$(curl -s -w '\t%{time_total}' -F file=@$f "http://127.0.0.1:$p/identify")
  t=${out##*$'\t'}; ct=$(printf '%s' "${out%$'\t'*}" | python3 -c 'import json,sys; d=json.load(sys.stdin); print((d.get("top1") or {}).get("ct_id","-"), d.get("detect_ms"), d.get("identify_ms"))' 2>/dev/null)
  printf '%s\t%s\t%s\t%s\n' $p "$(basename $f)" "$t" "$ct" >> /tmp/scanbench.tsv
done; done; done
python3 - <<'PY'
import collections, statistics as S
rows=[l.rstrip("\n").split("\t") for l in open("/tmp/scanbench.tsv")]
lat=collections.defaultdict(list); card=collections.defaultdict(dict)
for p,f,t,c in rows:
    lat[p].append(float(t)*1000); card[p][f]=c.split()[0]
q=lambda v,x: sorted(v)[min(len(v)-1,int(x*len(v)))]
names={"18197":"baseline (pt, as served)","18198":"pt + 4 torch threads + JPEG draft","18199":"onnx detector + JPEG draft"}
for p in sorted(lat): print(f"{names[p]:36s} n={len(lat[p])} p50={q(lat[p],.5):6.0f} p90={q(lat[p],.9):6.0f} p99={q(lat[p],.99):6.0f} mean={S.mean(lat[p]):6.0f} ms")
for p in ("18198","18199"):
    same=sum(card[p][f]==card["18197"][f] for f in card["18197"]); print(f"{names[p]}: same top-1 as baseline {same}/{len(card['18197'])}")
PY
