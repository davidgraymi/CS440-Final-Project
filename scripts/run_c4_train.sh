#!/usr/bin/env bash
# Long, resumable TRM training on the shared Mac mini, guarded so it can never
# starve the services on this machine -- above all AdGuard DNS, which the whole
# home network depends on.
#
# Safeguards:
#   - nice 19 (+ macOS background QoS unless QOS_BG=0); 2 CPU threads, 2 loader workers
#   - PyTorch MPS memory capped at MPS_MEM_RATIO of the GPU working set (0.3 ~= 3.2 GB)
#   - guard every 10 s:
#       * DNS failing DNS_FAIL_LIMIT (3) checks in a row, or slower than DNS_MAX_MS 3 in a row -> stop
#         (AdGuard drops ~3% of probes on its own, so 2 in a row would trip several times a day)
#       * AdGuardHome process missing -> stop
#       * free memory below HARD_MIN_FREE_PCT -> stop
#       * free memory below MIN_FREE_PCT for LOW_MEM_MINUTES in a row -> stop
#     "stop" = SIGTERM to the trainer (it saves a checkpoint after the current
#     step and exits), then SIGKILL after 20 s. Checkpoints are written
#     atomically, so even a SIGKILL leaves the last good one.
#   - refuses to start unless DNS is healthy and free memory >= MIN_FREE_PCT + 20
#
# Resuming: just run the script again with the same RUN_NAME; training continues
# from runs/$RUN_NAME/ckpt.pt (weights, EMA, optimizer, LR schedule, step).
#
# Usage (from the repo root):
#   nohup scripts/run_c4_train.sh > runs/c4-train-smallstep.log 2>&1 &
# Overrides: RUN_NAME=c4-trm-smallstep T=1 N=2 D_MODEL=256 BATCH=128 STEPS=1000000
#   EVAL_EVERY=2000 PATIENCE=10 TIME_LIMIT_MIN=0 EXTRA_ARGS= MPS_MEM_RATIO=0.3 TRAINER=c4|ponder
#   MIN_FREE_PCT=25 LOW_MEM_MINUTES=3 HARD_MIN_FREE_PCT=15 DNS_SERVER=192.168.0.253 DNS_MAX_MS=200 DNS_FAIL_LIMIT=3
set -uo pipefail
cd "$(dirname "$0")/.."

RUN_NAME=${RUN_NAME:-c4-trm-smallstep}
T=${T:-1}
N=${N:-2}
D_MODEL=${D_MODEL:-256}
BATCH=${BATCH:-128}
STEPS=${STEPS:-1000000}
EVAL_EVERY=${EVAL_EVERY:-2000}
PATIENCE=${PATIENCE:-10}
TIME_LIMIT_MIN=${TIME_LIMIT_MIN:-0}
EXTRA_ARGS=${EXTRA_ARGS:-}
MIN_FREE=${MIN_FREE_PCT:-25}
LOW_MEM_CHECKS=$(( ${LOW_MEM_MINUTES:-3} * 6 ))
HARD_MIN_FREE=${HARD_MIN_FREE_PCT:-15}
DNS_SERVER=${DNS_SERVER:-192.168.0.253}
DNS_MAX_MS=${DNS_MAX_MS:-200}
DNS_FAIL_LIMIT=${DNS_FAIL_LIMIT:-3}
GLOG=runs/$RUN_NAME-guard.log

export PYTORCH_MPS_HIGH_WATERMARK_RATIO=${MPS_MEM_RATIO:-0.3}
export PYTORCH_MPS_LOW_WATERMARK_RATIO=$(awk -v h="$PYTORCH_MPS_HIGH_WATERMARK_RATIO" 'BEGIN { print h * 0.75 }')
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
mkdir -p runs

dns_ms() { dig +tries=1 +time=2 @"$DNS_SERVER" example.com 2>/dev/null | awk '/Query time/ {print $4}'; }
free_pct() { memory_pressure 2>/dev/null | awk -F': ' '/free percentage/ {gsub("%", "", $2); print $2}'; }

pre_dns=$(dns_ms); pre_free=$(free_pct)
if [ -z "$pre_dns" ] || [ "$pre_dns" -gt "$DNS_MAX_MS" ] || [ -z "$pre_free" ] || [ "$pre_free" -lt $((MIN_FREE + 20)) ] \
   || ! pgrep -q AdGuardHome; then
  echo "$(date) | not starting: dns=${pre_dns:-FAIL} ms free=${pre_free:-?}% adguard=$(pgrep -q AdGuardHome && echo up || echo DOWN)" | tee -a "$GLOG"
  exit 1
fi

QOS=(taskpolicy -b); [ "${QOS_BG:-1}" = 0 ] && QOS=()
# Call train_main directly (works even if the venv's console scripts are stale).
# nice and taskpolicy exec the command, so $! is the trainer itself.
# TRAINER=ponder runs c4_ponder.py (outer-loop thinking, EXPERIMENT-OUTER-LOOP.md) instead of c4_train.py.
# shellcheck disable=SC2086
if [ "${TRAINER:-c4}" = ponder ]; then
  nice -n 19 "${QOS[@]}" .venv/bin/python -c "import sys; from cs440_final_project.c4_ponder import train_main; train_main(sys.argv[2:])" \
      c4-ponder --T "$T" --n "$N" --d-model "$D_MODEL" --batch-size "$BATCH" --steps "$STEPS" \
      --eval-every "$EVAL_EVERY" --time-limit-min "$TIME_LIMIT_MIN" --workers 2 --device mps \
      --run-name "$RUN_NAME" --resume $EXTRA_ARGS &
else
  nice -n 19 "${QOS[@]}" .venv/bin/python -c "import sys; from cs440_final_project.c4_train import train_main; train_main(sys.argv[2:])" \
      c4-train --model trm --T "$T" --n "$N" --no-halt --d-model "$D_MODEL" --batch-size "$BATCH" \
      --steps "$STEPS" --eval-every "$EVAL_EVERY" --eval-batch-size 128 --patience "$PATIENCE" \
      --time-limit-min "$TIME_LIMIT_MIN" --workers 2 --device mps --run-name "$RUN_NAME" --resume $EXTRA_ARGS &
fi
JOB=$!
caffeinate -i -w $JOB &

stop() {
  echo "$(date +%T) | STOP: $1" | tee -a "$GLOG"
  kill -TERM $JOB 2>/dev/null
  for _ in $(seq 20); do kill -0 $JOB 2>/dev/null || break; sleep 1; done
  if kill -0 $JOB 2>/dev/null; then
    echo "$(date +%T) | trainer still running after 20 s: SIGKILL" | tee -a "$GLOG"
    pkill -KILL -P $JOB 2>/dev/null; kill -KILL $JOB 2>/dev/null
  fi
  echo "$(date +%T) | stopped; resume by rerunning the same command (RUN_NAME=$RUN_NAME)" | tee -a "$GLOG"
  exit 2
}

echo "$(date +%T) | trainer pid $JOB ($RUN_NAME) started; guard: dns fail x${DNS_FAIL_LIMIT} or >${DNS_MAX_MS}ms x3," \
     "free<${HARD_MIN_FREE}% or <${MIN_FREE}% for ${LOW_MEM_CHECKS} checks, AdGuard gone (pre: dns ${pre_dns} ms, free ${pre_free}%)" | tee -a "$GLOG"
dns_fail=0 dns_slow=0 mem_low=0
while kill -0 $JOB 2>/dev/null; do
  free=$(free_pct); dns=$(dns_ms)
  if [ -z "$dns" ]; then dns_fail=$((dns_fail + 1)); dns_slow=0
  elif [ "$dns" -gt "$DNS_MAX_MS" ]; then dns_slow=$((dns_slow + 1)); dns_fail=0
  else dns_fail=0 dns_slow=0; fi
  if [ -n "$free" ] && [ "$free" -lt "$MIN_FREE" ]; then mem_low=$((mem_low + 1)); else mem_low=0; fi
  rss=$(ps -axo rss,command | awk '/train_main/ && !/awk/ {s += $1} END {printf "%d", s/1024}')
  echo "$(date +%T) | free ${free:-?}% | dns ${dns:-FAIL} ms | trainer rss ${rss} MB" >> "$GLOG"
  pgrep -q AdGuardHome || stop "AdGuardHome process not running"
  [ "$dns_fail" -ge "$DNS_FAIL_LIMIT" ] && stop "DNS failed $dns_fail checks in a row"
  [ "$dns_slow" -ge 3 ] && stop "DNS slower than ${DNS_MAX_MS} ms $dns_slow checks in a row"
  [ -n "$free" ] && [ "$free" -lt "$HARD_MIN_FREE" ] && stop "free memory ${free}% < ${HARD_MIN_FREE}%"
  [ "$mem_low" -ge "$LOW_MEM_CHECKS" ] && stop "free memory below ${MIN_FREE}% for $mem_low checks"
  sleep 10
done
wait $JOB; rc=$?
echo "$(date +%T) | trainer exited with code $rc" | tee -a "$GLOG"
exit $rc
