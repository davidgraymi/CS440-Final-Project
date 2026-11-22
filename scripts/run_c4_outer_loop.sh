#!/usr/bin/env bash
# EXPERIMENT-OUTER-LOOP.md end to end on this machine, unattended:
#   1. four pilots (2 x 2: target {move, per-column} x credit window {1, full}), ~2.5 h each
#   2. evaluate each pilot on the full tactical set; apply the section-4 go criteria
#   3. if a pilot passes: main run (winning settings, seeds 0 and 1), then fixed-steps controls
#      (K = 1, 4, 8, 24), each evaluated on the tactical and general sets
# Every training run goes through scripts/run_c4_train.sh (DNS / memory guard, resumable).
# Evaluations run under the same kind of guard. A guard stop waits 10 minutes and resumes,
# up to MAX_TRIES times per step. Rerunning this script skips finished steps.
#
# Usage (from the repo root):  nohup scripts/run_c4_outer_loop.sh >> runs/outer-loop.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."

PILOT_STEPS=${PILOT_STEPS:-2200}   # ~2.5 h at ~4.1 s/update (batch 128, 24 steps, compile + fp16)
MAIN_STEPS=${MAIN_STEPS:-16000}    # ~18 h, ~2M positions
CTRL_STEPS=${CTRL_STEPS:-16000}    # same positions as main; cost scales with K
MAX_TRIES=${MAX_TRIES:-8}
COMMON="--k-max 24 --beta 0.1 --prior-steps 6 --compile --amp fp16"
OUT=results/c4p
LOG=runs/outer-loop-orchestrator.log
DNS_SERVER=${DNS_SERVER:-192.168.0.253}
mkdir -p runs "$OUT"

log() { echo "$(date '+%F %T') | $*" | tee -a "$LOG"; }

train() {  # name steps extra-args...
  local name=$1 steps=$2; shift 2
  if [ -f "runs/$name/done.json" ]; then log "$name: already done"; return 0; fi
  for try in $(seq "$MAX_TRIES"); do
    log "$name: training (try $try), steps $steps, args: $*"
    TRAINER=ponder RUN_NAME=$name T=1 N=1 STEPS=$steps EVAL_EVERY=250 TIME_LIMIT_MIN=0 \
      EXTRA_ARGS="$COMMON $*" scripts/run_c4_train.sh >> "runs/$name.log" 2>&1
    local rc=$?
    if [ -f "runs/$name/done.json" ]; then
      mkdir -p "$OUT/$name" && cp "runs/$name/log.jsonl" "runs/$name/config.json" "$OUT/$name/"
      log "$name: done"; return 0
    fi
    log "$name: runner exited with $rc before finishing; waiting 10 min, then resuming"
    sleep 600
  done
  log "$name: gave up after $MAX_TRIES tries"; return 1
}

dns_ms() { dig +tries=1 +time=2 @"$DNS_SERVER" example.com 2>/dev/null | awk '/Query time/ {print $4}'; }
free_pct() { memory_pressure 2>/dev/null | awk -F': ' '/free percentage/ {gsub("%", "", $2); print $2}'; }

guarded() {  # run a python callable string with args under the DNS / memory guard; 0 ok, 2 stopped
  local code=$1; shift
  PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.3 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.225 OMP_NUM_THREADS=2 \
    nice -n 19 taskpolicy -b .venv/bin/python -c "$code" c4-guarded-eval "$@" >> "$LOG" 2>&1 &
  local job=$! fails=0
  while kill -0 $job 2>/dev/null; do
    local dns free; dns=$(dns_ms); free=$(free_pct)
    if [ -z "$dns" ] || [ "$dns" -gt 200 ]; then fails=$((fails + 1)); else fails=0; fi
    if [ $fails -ge 3 ] || ! pgrep -q AdGuardHome || { [ -n "$free" ] && [ "$free" -lt 25 ]; }; then
      log "eval guard STOP: dns fails $fails, free ${free:-?}%"; kill -TERM $job; sleep 5; kill -KILL $job 2>/dev/null
      return 2
    fi
    sleep 10
  done
  wait $job
}

evaluate() {  # name split bootstrap
  local name=$1 split=$2 boot=$3 out="$OUT/$1/eval-$2.json"
  if [ -f "$out" ]; then return 0; fi
  mkdir -p "$OUT/$name"
  for try in $(seq "$MAX_TRIES"); do
    log "$name: evaluating on $split (try $try)"
    guarded "import sys; from cs440_final_project.c4_ponder import eval_main; eval_main(sys.argv[2:])" \
      --ckpt "runs/$name/ckpt.pt" --split "$split" --bootstrap "$boot" --device mps --out "$out" && [ -f "$out" ] && return 0
    sleep 600
  done
  log "$name: eval on $split failed"; return 1
}

flags() {  # pilot name -> its settings (bash 3.2 on macOS has no associative arrays)
  case $1 in
    c4p-pilot-D) echo "--window 0 --col-weight 1" ;;  # full credit, per-column target (full design)
    c4p-pilot-C) echo "--window 0 --col-weight 0" ;;  # full credit, move target only
    c4p-pilot-B) echo "--window 1 --col-weight 1" ;;  # no credit across steps, per-column target
    c4p-pilot-A) echo "--window 1 --col-weight 0" ;;  # no credit across steps, move only (control)
  esac
}

log "=== outer-loop experiment: pilots ==="
evals=()
for name in c4p-pilot-D c4p-pilot-C c4p-pilot-B c4p-pilot-A; do
  train "$name" "$PILOT_STEPS" $(flags "$name") --seed 0 || continue
  evaluate "$name" tactical 200 && evals+=("$name=$OUT/$name/eval-tactical.npz")
done

winner=$(.venv/bin/python -c "import sys; from cs440_final_project.c4_ponder import decide_main; decide_main(sys.argv[2:])" \
         c4-decide --evals "${evals[@]}" --out "$OUT/pilot-decision.json" 2>>"$LOG" | tail -1)
log "pilot decision: winner = ${winner:-none} (details: $OUT/pilot-decision.json)"
if [ -z "$winner" ] || [ "$winner" = none ]; then
  log "no pilot passed the go criteria; stopping here (no multi-day run)"; exit 0
fi

WIN=$(flags "$winner")
log "=== main run with $winner settings: $WIN ==="
for seed in 0 1; do
  name=c4p-main-s$seed
  train "$name" "$MAIN_STEPS" $WIN --seed $seed && evaluate "$name" tactical 1000 && evaluate "$name" general 200
done
log "=== fixed-steps controls ==="
for k in 1 4 8 24; do
  name=c4p-fixed-k$k
  train "$name" "$CTRL_STEPS" $WIN --fixed-steps $k --seed 0 && evaluate "$name" tactical 1000 && evaluate "$name" general 200
done
log "=== all done ==="
