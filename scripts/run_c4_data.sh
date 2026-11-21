#!/usr/bin/env bash
# Build the v2 Connect Four dataset (c4data.build_main) on the shared Mac mini,
# guarded so the solver can never starve AdGuard DNS.
#
# Stage 0 measured 2 background workers at ~100 positions/s with DNS failures at
# 1.1% (AdGuard's own baseline: 2.9%). See EXPERIMENT.md, section 3.5.
#
# Safeguards:
#   - nice 19 + macOS background QoS (efficiency cores only); WORKERS solver processes (default 2)
#   - guard every 10 s: stop on DNS_FAIL_LIMIT (3) failed probes in a row, 3 probes in a row
#     slower than DNS_MAX_MS, AdGuardHome missing, or free memory below MIN_FREE_PCT
#   - stop = SIGTERM to the whole process tree (Python workers and c4solver), SIGKILL after 10 s
#   - resumable: finished shards are kept; rerun the same command to continue
#
# Usage (from the repo root):
#   nohup scripts/run_c4_data.sh > runs/c4data-v2.log 2>&1 &
# Overrides: OUT=runs/c4data-v2 SHARDS=500 PER_SHARD=5000 WORKERS=2 MIN_FREE_PCT=30
#   DNS_SERVER=192.168.0.253 DNS_MAX_MS=200 DNS_FAIL_LIMIT=3
set -uo pipefail
cd "$(dirname "$0")/.."

OUT=${OUT:-runs/c4data-v2}
SHARDS=${SHARDS:-500}
PER_SHARD=${PER_SHARD:-5000}
WORKERS=${WORKERS:-2}
MIN_FREE=${MIN_FREE_PCT:-30}
DNS_SERVER=${DNS_SERVER:-192.168.0.253}
DNS_MAX_MS=${DNS_MAX_MS:-200}
DNS_FAIL_LIMIT=${DNS_FAIL_LIMIT:-3}
GLOG=$OUT-guard.log
mkdir -p "$(dirname "$OUT")"
export OMP_NUM_THREADS=1

dns_ms() { dig +tries=1 +time=2 @"$DNS_SERVER" example.com 2>/dev/null | awk '/Query time/ {print $4}'; }
free_pct() { memory_pressure 2>/dev/null | awk -F': ' '/free percentage/ {gsub("%", "", $2); print $2}'; }
tree() { local p; for p in $(pgrep -P "$1"); do echo "$p"; tree "$p"; done; }

pre_dns=$(dns_ms); pre_free=$(free_pct)
if [ -z "$pre_dns" ] || [ "$pre_dns" -gt "$DNS_MAX_MS" ] || [ -z "$pre_free" ] || [ "$pre_free" -lt $((MIN_FREE + 20)) ] \
   || ! pgrep -q AdGuardHome; then
  echo "$(date) | not starting: dns=${pre_dns:-FAIL} ms free=${pre_free:-?}% adguard=$(pgrep -q AdGuardHome && echo up || echo DOWN)" | tee -a "$GLOG"
  exit 1
fi

nice -n 19 taskpolicy -b .venv/bin/python -c "import sys; from cs440_final_project.c4data import build_main; build_main(sys.argv[2:])" \
    c4-guarded-data --out "$OUT" --shards "$SHARDS" --positions-per-shard "$PER_SHARD" --workers "$WORKERS" &
JOB=$!
caffeinate -i -w $JOB &

stop() {
  echo "$(date +%T) | STOP: $1" | tee -a "$GLOG"
  local pids; pids="$JOB $(tree $JOB)"
  kill -TERM $pids 2>/dev/null
  for _ in $(seq 10); do kill -0 $JOB 2>/dev/null || break; sleep 1; done
  kill -KILL $pids 2>/dev/null; pkill -KILL -f "c4solver -a" 2>/dev/null
  echo "$(date +%T) | stopped; finished shards are kept. Resume with: nohup scripts/run_c4_data.sh >> $OUT.log 2>&1 &" | tee -a "$GLOG"
  exit 2
}

echo "$(date +%T) | data build pid $JOB started ($SHARDS shards x $PER_SHARD, $WORKERS workers); guard: dns fail x$DNS_FAIL_LIMIT" \
     "or >${DNS_MAX_MS}ms x3, free<${MIN_FREE}%, AdGuard gone (pre: dns ${pre_dns} ms, free ${pre_free}%)" | tee -a "$GLOG"
dns_fail=0 dns_slow=0
while kill -0 $JOB 2>/dev/null; do
  free=$(free_pct); dns=$(dns_ms)
  if [ -z "$dns" ]; then dns_fail=$((dns_fail + 1)); dns_slow=0
  elif [ "$dns" -gt "$DNS_MAX_MS" ]; then dns_slow=$((dns_slow + 1)); dns_fail=0
  else dns_fail=0 dns_slow=0; fi
  shards=$(ls "$OUT"/shard_*[0-9].npz 2>/dev/null | wc -l | tr -d ' ')
  echo "$(date +%T) | free ${free:-?}% | dns ${dns:-FAIL} ms | shards $shards/$SHARDS" >> "$GLOG"
  pgrep -q AdGuardHome || stop "AdGuardHome process not running"
  [ "$dns_fail" -ge "$DNS_FAIL_LIMIT" ] && stop "DNS failed $dns_fail checks in a row"
  [ "$dns_slow" -ge 3 ] && stop "DNS slower than ${DNS_MAX_MS} ms $dns_slow checks in a row"
  [ -n "$free" ] && [ "$free" -lt "$MIN_FREE" ] && stop "free memory ${free}% < ${MIN_FREE}%"
  sleep 10
done
wait $JOB; rc=$?
echo "$(date +%T) | data build exited with code $rc" | tee -a "$GLOG"
exit $rc
