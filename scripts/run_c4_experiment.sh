#!/usr/bin/env bash
# Connect Four experiment on a shared Mac (Apple silicon, MPS) that must finish
# by a deadline, keeping headroom for services on the same machine (AdGuard,
# Traefik, Uptime Kuma):
#   - everything runs at the lowest CPU priority (nice 19)
#   - CPU threads / workers leave RESERVE_CORES cores free
#   - PyTorch's GPU memory is capped (MPS shares RAM with everything else)
#   - a guard logs free memory and DNS latency every 30 s and stops the
#     experiment if free memory drops below MIN_FREE_PCT, or if DNS fails or
#     is slower than DNS_MAX_MS for DNS_BAD_LIMIT checks in a row
#   - the run refuses to start if DNS is not healthy beforehand
#   - the Mac is kept awake (caffeinate) while it runs
#
# Schedule: build data if needed, then train TRM for most of the time left
# before DEADLINE. ds-only and direct then train for the SAME number of
# optimizer steps TRM reached (same data and updates, less compute per step),
# each capped to its share of the remaining time. Evaluations run last.
#
# Usage (from the repo root):
#   mkdir -p runs && nohup scripts/run_c4_experiment.sh > runs/c4-experiment.log 2>&1 &
#
# Overrides (environment): DEADLINE=07:00 RESERVE_CORES=2 MPS_MEM_RATIO=0.4
#   MIN_FREE_PCT=25 DNS_SERVER=192.168.0.253 DNS_MAX_MS=300 DNS_BAD_LIMIT=3
#   DATA_WORKERS=2 D_MODEL=256 BATCH=128 TAG=overnight
set -euo pipefail
cd "$(dirname "$0")/.."

DEADLINE=${DEADLINE:-07:00}
RESERVE=${RESERVE_CORES:-2}
TOTAL=$(sysctl -n hw.ncpu)
USE=$((TOTAL - RESERVE))
if [ "$USE" -lt 1 ]; then echo "need more than $RESERVE cores (have $TOTAL)"; exit 1; fi
THREADS=$(( USE < 4 ? USE : 4 ))
LOADERS=2
MIN_FREE=${MIN_FREE_PCT:-25}
DNS_SERVER=${DNS_SERVER:-192.168.0.253}
DNS_MAX_MS=${DNS_MAX_MS:-300}
DNS_BAD_LIMIT=${DNS_BAD_LIMIT:-3}
DATA_WORKERS=${DATA_WORKERS:-2}
D_MODEL=${D_MODEL:-256}
BATCH=${BATCH:-128}
TAG=${TAG:-overnight}
EVAL_RESERVE_MIN=40   # final evaluations of all three models

# PyTorch MPS allocations stop at this fraction of the GPU's recommended working
# set (~10.7 GB on a 16 GB Mac): 0.4 ~= 4.3 GB.
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=${MPS_MEM_RATIO:-0.4}
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
run() { nice -n 19 "$@"; }
mkdir -p runs

deadline_s=$(date -j -f "%Y-%m-%d %H:%M" "$(date +%F) $DEADLINE" +%s)
[ "$deadline_s" -le "$(date +%s)" ] && deadline_s=$((deadline_s + 86400))
minutes_left() { echo $(( (deadline_s - $(date +%s)) / 60 - EVAL_RESERVE_MIN )); }
share() { awk -v m="$(minutes_left)" -v f="$1" 'BEGIN { v = m * f; print (v > 5 ? int(v) : 5) }'; }

dns_ms() {  # query time in ms, empty on failure
  dig +tries=1 +time=2 @"$DNS_SERVER" example.com 2>/dev/null | awk '/Query time/ {print $4}' || true
}
stop_all() {
  echo "$(date) | $1: stopping the experiment" | tee -a runs/c4-guard.log
  pkill -TERM -f "c4-(train|eval|data)" || true
  pkill -TERM -f c4solver || true
  kill -TERM $$
  exit 0
}

pre=$(dns_ms)
if [ -z "$pre" ] || [ "$pre" -gt "$DNS_MAX_MS" ]; then
  echo "$(date) | DNS @$DNS_SERVER unhealthy before starting (${pre:-no answer}); not starting" | tee -a runs/c4-guard.log
  exit 1
fi

guard() {
  set +e +o pipefail  # a failed probe must not kill the guard
  echo "$(date +%T) | guard started (DNS @$DNS_SERVER ${pre} ms; stop on ${DNS_BAD_LIMIT}x >${DNS_MAX_MS} ms or fail, or < ${MIN_FREE}% free)" >> runs/c4-guard.log
  bad=0
  while sleep 30; do
    free=$(memory_pressure 2>/dev/null | awk -F': ' '/free percentage/ {gsub("%", "", $2); print $2}')
    dns=$(dns_ms)
    if [ -z "$dns" ] || [ "$dns" -gt "$DNS_MAX_MS" ]; then bad=$((bad + 1)); else bad=0; fi
    echo "$(date +%T) | mem free ${free:-?}% | dns ${dns:-FAIL} ms | bad ${bad}/${DNS_BAD_LIMIT}" >> runs/c4-guard.log
    if [ -n "$free" ] && [ "$free" -lt "$MIN_FREE" ]; then stop_all "memory free ${free}% < ${MIN_FREE}%"; fi
    if [ "$bad" -ge "$DNS_BAD_LIMIT" ]; then stop_all "DNS slow or failing ${bad} checks in a row"; fi
  done
}
guard &
GUARD=$!
caffeinate -i -w $$ &
trap 'kill $GUARD 2>/dev/null || true' EXIT

echo "$(date) | deadline $(date -r "$deadline_s") | $TOTAL cores, using $USE (threads $THREADS)" \
     "| MPS mem ratio $PYTORCH_MPS_HIGH_WATERMARK_RATIO | d_model $D_MODEL batch $BATCH" \
     "| guard: free >= ${MIN_FREE}%, DNS @$DNS_SERVER"

run uv sync
run scripts/setup_connect4_solver.sh
if [ ! -f data/c4_test.npz ]; then
  echo "$(date) | building dataset"
  run uv run c4-data --num 1000000 --test-per-bucket 2000 --workers "$DATA_WORKERS"
fi

train() {  # train <model> <max steps> <minutes>
  local name="c4-$1-$TAG"
  if [ -f "runs/$name/done.json" ]; then echo "$(date) | $name already done"; return; fi
  echo "$(date) | training $1 for up to $3 min / $2 steps"
  rm -rf "runs/$name"
  run uv run c4-train --model "$1" --d-model "$D_MODEL" --batch-size "$BATCH" --steps "$2" \
      --time-limit-min "$3" --eval-every 2000 --workers "$LOADERS" --device mps --run-name "$name"
}

train trm 1000000 "$(share 0.75)"
TRM_STEPS=$(uv run python -c "import json; print(json.load(open('runs/c4-trm-$TAG/done.json'))['step'])")
echo "$(date) | TRM reached $TRM_STEPS steps; matching that for the baselines"
train ds-only "$TRM_STEPS" "$(share 0.65)"
train direct "$TRM_STEPS" "$(share 0.9)"

for m in trm ds-only direct; do
  out="runs/c4eval-c4-$m-$TAG"
  if [ ! -f "$out/summary.json" ]; then
    echo "$(date) | evaluating $m"
    run uv run c4-eval --ckpt "runs/c4-$m-$TAG/ckpt.pt" --per-bucket 1000 --device mps --batch-size 512 --out "$out"
  fi
done
echo "$(date) | done (deadline $(date -r "$deadline_s"))"
