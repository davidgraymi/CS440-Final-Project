#!/usr/bin/env bash
# Test-time recursion sweep (c4-sweep) on the shared Mac mini, guarded so it can
# never starve the services on this machine -- above all AdGuard DNS, which the
# whole home network depends on.
#
# Safeguards (stricter than run_c4_experiment.sh, since this is inference only):
#   - macOS background QoS (taskpolicy -b) + nice 19: lowest CPU and I/O priority
#   - 2 CPU threads; PyTorch MPS memory capped at MPS_MEM_RATIO (~1.6 GB default)
#   - guard every 10 s, DNS stop always on:
#       * DNS failing 2 checks in a row, or slower than DNS_MAX_MS 3 in a row -> stop
#       * AdGuardHome process missing -> stop
#       * free memory below MIN_FREE_PCT -> stop immediately
#     "stop" = SIGTERM, then SIGKILL after 5 s.
#
# Usage (from the repo root):
#   nohup scripts/run_c4_sweep.sh > runs/c4-sweep.log 2>&1 &
# Overrides: CKPT PER_BUCKET=500 T_LIST N_LIST BATCH=128 OUT MPS_MEM_RATIO=0.15
#   MIN_FREE_PCT=40 DNS_SERVER=192.168.0.253 DNS_MAX_MS=200
set -uo pipefail
cd "$(dirname "$0")/.."

CKPT=${CKPT:-results/c4-overnight/c4-trm-overnight/ckpt.pt}
PER_BUCKET=${PER_BUCKET:-500}
T_LIST=${T_LIST:-1,2,3,4,6}
N_LIST=${N_LIST:-1,2,3,6,9}
BATCH=${BATCH:-128}
OUT=${OUT:-runs/c4sweep-c4-trm-overnight-mps}
MIN_FREE=${MIN_FREE_PCT:-40}
DNS_SERVER=${DNS_SERVER:-192.168.0.253}
DNS_MAX_MS=${DNS_MAX_MS:-200}
GLOG=runs/c4-sweep-guard.log

export PYTORCH_MPS_HIGH_WATERMARK_RATIO=${MPS_MEM_RATIO:-0.15}
export PYTORCH_MPS_LOW_WATERMARK_RATIO=$(awk -v h="$PYTORCH_MPS_HIGH_WATERMARK_RATIO" 'BEGIN { print h * 0.75 }')
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
mkdir -p runs

dns_ms() { dig +tries=1 +time=2 @"$DNS_SERVER" example.com 2>/dev/null | awk '/Query time/ {print $4}'; }
free_pct() { memory_pressure 2>/dev/null | awk -F': ' '/free percentage/ {gsub("%", "", $2); print $2}'; }

# Refuse to start unless the machine is healthy right now.
pre_dns=$(dns_ms); pre_free=$(free_pct)
if [ -z "$pre_dns" ] || [ "$pre_dns" -gt "$DNS_MAX_MS" ] || [ -z "$pre_free" ] || [ "$pre_free" -lt $((MIN_FREE + 15)) ] \
   || ! pgrep -q AdGuardHome; then
  echo "$(date) | not starting: dns=${pre_dns:-FAIL} ms free=${pre_free:-?}% adguard=$(pgrep -q AdGuardHome && echo up || echo DOWN)" | tee -a "$GLOG"
  exit 1
fi

# Call sweep_main directly (works even if the venv's console scripts are stale).
nice -n 19 taskpolicy -b .venv/bin/python -c "import sys; from cs440_final_project.c4_train import sweep_main; sweep_main(sys.argv[2:])" \
    c4-sweep --ckpt "$CKPT" --per-bucket "$PER_BUCKET" \
    --T "$T_LIST" --n "$N_LIST" --batch-size "$BATCH" --device mps --out "$OUT" &
JOB=$!
caffeinate -i -w $JOB &

stop() {
  echo "$(date +%T) | STOP: $1" | tee -a "$GLOG"
  pkill -TERM -P $JOB 2>/dev/null; kill -TERM $JOB 2>/dev/null; pkill -TERM -f "c4-sweep" 2>/dev/null
  sleep 5
  pkill -KILL -P $JOB 2>/dev/null; kill -KILL $JOB 2>/dev/null; pkill -KILL -f "c4-sweep" 2>/dev/null
  exit 2
}

echo "$(date +%T) | sweep pid $JOB started; guard: dns fail x2 or >${DNS_MAX_MS}ms x3, free<${MIN_FREE}%, AdGuard gone (pre: dns ${pre_dns} ms, free ${pre_free}%)" | tee -a "$GLOG"
dns_fail=0 dns_slow=0
while kill -0 $JOB 2>/dev/null; do
  free=$(free_pct); dns=$(dns_ms)
  if [ -z "$dns" ]; then dns_fail=$((dns_fail + 1)); dns_slow=0
  elif [ "$dns" -gt "$DNS_MAX_MS" ]; then dns_slow=$((dns_slow + 1)); dns_fail=0
  else dns_fail=0 dns_slow=0; fi
  rss=$(ps -axo rss,command | awk '/c4-sweep/ && !/awk/ && !/run_c4_sweep/ {s += $1} END {printf "%d", s/1024}')
  echo "$(date +%T) | free ${free:-?}% | dns ${dns:-FAIL} ms | sweep rss ${rss} MB" >> "$GLOG"
  pgrep -q AdGuardHome || stop "AdGuardHome process not running"
  [ "$dns_fail" -ge 2 ] && stop "DNS failed $dns_fail checks in a row"
  [ "$dns_slow" -ge 3 ] && stop "DNS slower than ${DNS_MAX_MS} ms $dns_slow checks in a row"
  [ -n "$free" ] && [ "$free" -lt "$MIN_FREE" ] && stop "free memory ${free}% < ${MIN_FREE}%"
  sleep 10
done
wait $JOB; rc=$?
echo "$(date +%T) | sweep exited with code $rc" | tee -a "$GLOG"
exit $rc
