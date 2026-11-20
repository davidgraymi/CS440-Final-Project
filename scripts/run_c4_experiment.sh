#!/usr/bin/env bash
# Full Connect Four experiment on a shared Mac (Apple silicon, MPS), keeping
# headroom for other services (AdGuard DNS in Docker):
#   - everything runs at the lowest CPU priority (nice 19)
#   - CPU threads / workers leave RESERVE_CORES cores free
#   - PyTorch's GPU memory is capped (MPS shares RAM with the CPU and Docker)
#   - a guard logs free memory and DNS latency every 30 s and stops the
#     experiment if free memory drops below MIN_FREE_PCT
#   - the Mac is kept awake (caffeinate) while it runs
#
# Usage (from the repo root):
#   mkdir -p runs && nohup scripts/run_c4_experiment.sh > runs/c4-experiment.log 2>&1 &
#
# Overrides (environment): RESERVE_CORES=2 MPS_MEM_RATIO=0.4 MIN_FREE_PCT=25
#   D_MODEL=256 BATCH=128 STEPS=30000 MODELS="trm ds-only direct" DNS_SERVER=127.0.0.1
# Re-running skips finished stages (dataset, finished trainings, evals);
# an unfinished training restarts from scratch.
set -euo pipefail
cd "$(dirname "$0")/.."

RESERVE=${RESERVE_CORES:-2}
TOTAL=$(sysctl -n hw.ncpu)
USE=$((TOTAL - RESERVE))
if [ "$USE" -lt 1 ]; then echo "need more than $RESERVE cores (have $TOTAL)"; exit 1; fi
THREADS=$(( USE < 4 ? USE : 4 ))
LOADERS=2
MIN_FREE=${MIN_FREE_PCT:-25}
DNS_SERVER=${DNS_SERVER:-127.0.0.1}
D_MODEL=${D_MODEL:-256}
BATCH=${BATCH:-128}
STEPS=${STEPS:-30000}
MODELS=${MODELS:-"trm ds-only direct"}

# PyTorch MPS allocations stop at this fraction of the GPU's recommended working
# set (~10.7 GB on a 16 GB Mac): 0.4 ~= 4.3 GB. Raise it only if training OOMs
# and memory headroom allows.
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=${MPS_MEM_RATIO:-0.4}
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
run() { nice -n 19 "$@"; }
mkdir -p runs

guard() {
  while sleep 30; do
    free=$(memory_pressure | awk -F': ' '/free percentage/ {gsub("%", "", $2); print $2}')
    dns=$(dig +tries=1 +time=2 @"$DNS_SERVER" example.com 2>/dev/null | awk '/Query time/ {print $4}')
    echo "$(date +%T) | mem free ${free:-?}% | dns ${dns:-FAIL} ms" >> runs/c4-guard.log
    if [ -n "$free" ] && [ "$free" -lt "$MIN_FREE" ]; then
      echo "$(date) | memory free ${free}% < ${MIN_FREE}%: stopping the experiment" | tee -a runs/c4-guard.log
      pkill -TERM -f "c4-(train|eval|data)" || true
      kill -TERM $$
      exit 0
    fi
  done
}
guard &
GUARD=$!
caffeinate -i -w $$ &
trap 'kill $GUARD 2>/dev/null || true' EXIT

echo "$(date) | $TOTAL cores, using $USE (threads $THREADS) | MPS mem ratio $PYTORCH_MPS_HIGH_WATERMARK_RATIO" \
     "| d_model $D_MODEL batch $BATCH steps $STEPS | guard: free >= ${MIN_FREE}%, DNS @$DNS_SERVER"

run uv sync
run scripts/setup_connect4_solver.sh
if [ ! -f data/c4_test.npz ]; then
  echo "$(date) | building dataset"
  run uv run c4-data --num 1000000 --test-per-bucket 2000 --workers "$USE"
fi

for m in $MODELS; do
  if ! grep -q "\"step\": $STEPS, \"eval\"" "runs/c4-$m/log.jsonl" 2>/dev/null; then
    echo "$(date) | training $m"
    rm -rf "runs/c4-$m"
    run uv run c4-train --model "$m" --d-model "$D_MODEL" --batch-size "$BATCH" --steps "$STEPS" \
        --workers "$LOADERS" --device mps --run-name "c4-$m"
  fi
  if [ ! -f "runs/c4eval-c4-$m/summary.json" ]; then
    echo "$(date) | evaluating $m"
    run uv run c4-eval --ckpt "runs/c4-$m/ckpt.pt" --device mps --batch-size 512 --out "runs/c4eval-c4-$m"
  fi
done

# Does thinking longer than trained help? (TRM only)
if [[ " $MODELS " == *" trm "* ]] && [ ! -f runs/c4eval-c4-trm-steps32/summary.json ]; then
  run uv run c4-eval --ckpt runs/c4-trm/ckpt.pt --steps 32 --device mps --batch-size 512 \
      --out runs/c4eval-c4-trm-steps32
fi
echo "$(date) | done"
