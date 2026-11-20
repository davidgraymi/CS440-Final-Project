#!/usr/bin/env bash
# Full Connect Four experiment on a shared Linux server, leaving headroom for
# other services (e.g. DNS):
#   - CPU cores 0..RESERVE_CORES-1 are left untouched (taskset pins us to the rest)
#   - everything runs at the lowest CPU and I/O priority (nice 19, ionice idle)
#   - only the GPUs listed in GPUS are used
#
# Usage (from the repo root):
#   nohup scripts/run_c4_experiment.sh > runs/c4-experiment.log 2>&1 &
#
# Environment overrides: RESERVE_CORES (default 2), GPUS (default 0),
# STEPS (default 50000), MODELS (default "trm ds-only direct").
# Re-running skips finished stages (dataset, trained checkpoints, evals).
set -euo pipefail
cd "$(dirname "$0")/.."

RESERVE=${RESERVE_CORES:-2}
TOTAL=$(nproc --all)
USE=$((TOTAL - RESERVE))
if [ "$USE" -lt 1 ]; then echo "need more than $RESERVE cores (have $TOTAL)"; exit 1; fi
CPUS="$RESERVE-$((TOTAL - 1))"
LOADERS=$(( USE < 4 ? USE : 4 ))
STEPS=${STEPS:-50000}
MODELS=${MODELS:-"trm ds-only direct"}

export CUDA_VISIBLE_DEVICES=${GPUS:-0}
export OMP_NUM_THREADS=$USE MKL_NUM_THREADS=$USE
run() { nice -n 19 ionice -c 3 taskset -c "$CPUS" "$@"; }

echo "$(date) | cores $CPUS of $TOTAL (reserving 0-$((RESERVE - 1))) | GPUs $CUDA_VISIBLE_DEVICES | steps $STEPS"
mkdir -p runs

run uv sync
run scripts/setup_connect4_solver.sh
if [ ! -f data/c4_test.npz ]; then
  run uv run c4-data --num 1000000 --test-per-bucket 2000 --workers "$USE"
fi

for m in $MODELS; do
  if [ ! -f "runs/c4-$m/ckpt.pt" ] || ! grep -q "\"step\": $STEPS, \"eval\"" "runs/c4-$m/log.jsonl" 2>/dev/null; then
    echo "$(date) | training $m"
    rm -rf "runs/c4-$m"
    run uv run c4-train --model "$m" --steps "$STEPS" --compile --workers "$LOADERS" --run-name "c4-$m"
  fi
  if [ ! -f "runs/c4eval-c4-$m/summary.json" ]; then
    echo "$(date) | evaluating $m"
    run uv run c4-eval --ckpt "runs/c4-$m/ckpt.pt" --out "runs/c4eval-c4-$m"
  fi
done

# Does thinking longer than trained help? (TRM only)
if [[ " $MODELS " == *" trm "* ]] && [ ! -f runs/c4eval-c4-trm-steps32/summary.json ]; then
  run uv run c4-eval --ckpt runs/c4-trm/ckpt.pt --steps 32 --out runs/c4eval-c4-trm-steps32
fi
echo "$(date) | done"
