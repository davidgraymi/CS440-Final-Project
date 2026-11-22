#!/usr/bin/env bash
# Run ONE command under the same safeguards as run_c4_train.sh. The job queue
# (c4-queue, src/cs440_final_project/jobqueue.py) wraps every job in this, on
# whichever Mac the job lands on.
#
# Safeguards:
#   - nice 19 (+ macOS background QoS when QOS_BG=1); THREADS CPU threads
#   - PyTorch MPS memory capped at MPS_MEM_RATIO of the GPU working set
#   - the Mac is kept awake while the job runs (caffeinate -i -s; -s only
#     holds while on AC power, and a MacBook still sleeps when the lid closes)
#   - guard every 10 s:
#       * free memory below HARD_MIN_FREE_PCT -> stop
#       * free memory below MIN_FREE_PCT for LOW_MEM_MINUTES in a row -> stop
#       * the queue worker that launched this job is gone (C4_WATCH_PID) -> stop,
#         so a job never outlives its Ray node (e.g. after `ray stop`)
#       * only with C4_GUARD_DNS=1 (the Mac mini; set by `ray_node.sh head`):
#         DNS failing DNS_FAIL_LIMIT checks in a row, or slower than DNS_MAX_MS
#         3 in a row, or the AdGuardHome process missing -> stop
#     "stop" = SIGTERM to the job's process group (c4-train saves a checkpoint
#     after the current step and exits), then SIGKILL after STOP_GRACE_S.
#   - refuses to start (exit 75) unless free memory >= MIN_FREE_PCT + 20 and,
#     with C4_GUARD_DNS=1, DNS and AdGuard are healthy
#
# Usage (from the repo root):  scripts/guarded_run.sh <command> [args...]
# Overrides: C4_GUARD_DNS=0 QOS_BG=$C4_GUARD_DNS MPS_MEM_RATIO=0.3 THREADS=2
#   MIN_FREE_PCT=25 LOW_MEM_MINUTES=3 HARD_MIN_FREE_PCT=15 STOP_GRACE_S=20
#   DNS_SERVER=192.168.0.253 DNS_MAX_MS=200 DNS_FAIL_LIMIT=3 C4_WATCH_PID=
# Exit code: the command's own, 2 if the guard stopped it, 75 if it refused to start.
set -uo pipefail
set -m   # background jobs get their own process group, so the guard can signal all of it
[ $# -ge 1 ] || { echo "usage: $0 <command> [args...]" >&2; exit 64; }

GUARD_DNS=${C4_GUARD_DNS:-0}
QOS_BG=${QOS_BG:-$GUARD_DNS}
THREADS=${THREADS:-2}
MIN_FREE=${MIN_FREE_PCT:-25}
LOW_MEM_CHECKS=$(( ${LOW_MEM_MINUTES:-3} * 6 ))
HARD_MIN_FREE=${HARD_MIN_FREE_PCT:-15}
STOP_GRACE_S=${STOP_GRACE_S:-20}
DNS_SERVER=${DNS_SERVER:-192.168.0.253}
DNS_MAX_MS=${DNS_MAX_MS:-200}
DNS_FAIL_LIMIT=${DNS_FAIL_LIMIT:-3}
WATCH_PID=${C4_WATCH_PID:-}

export PYTORCH_MPS_HIGH_WATERMARK_RATIO=${MPS_MEM_RATIO:-0.3}
export PYTORCH_MPS_LOW_WATERMARK_RATIO=$(awk -v h="$PYTORCH_MPS_HIGH_WATERMARK_RATIO" 'BEGIN { print h * 0.75 }')
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS

say() { echo "$(date '+%F %T') | guard | $*"; }
dns_ms() { dig +tries=1 +time=2 @"$DNS_SERVER" example.com 2>/dev/null | awk '/Query time/ {print $4}'; }
free_pct() { memory_pressure 2>/dev/null | awk -F': ' '/free percentage/ {gsub("%", "", $2); print $2}'; }
adguard() { pgrep -q AdGuardHome && echo up || echo DOWN; }

# Refuse to start unless the machine is healthy right now.
pre_free=$(free_pct)
if [ -z "$pre_free" ] || [ "$pre_free" -lt $((MIN_FREE + 20)) ]; then
  say "not starting: free memory ${pre_free:-?}% < $((MIN_FREE + 20))%"; exit 75
fi
pre_dns=-
if [ "$GUARD_DNS" = 1 ]; then
  pre_dns=$(dns_ms)
  if [ -z "$pre_dns" ] || [ "$pre_dns" -gt "$DNS_MAX_MS" ] || [ "$(adguard)" = DOWN ]; then
    say "not starting: dns=${pre_dns:-FAIL} ms adguard=$(adguard)"; exit 75
  fi
fi

QOS=(); [ "$QOS_BG" = 1 ] && QOS=(taskpolicy -b)
# ${QOS[@]+...}: macOS bash 3.2 treats an empty array as unset under set -u
nice -n 19 ${QOS[@]+"${QOS[@]}"} "$@" &
JOB=$!   # nice/taskpolicy exec the command; with set -m it leads its own process group
caffeinate -i -s -w $JOB &

stop() {
  say "STOP: $1"
  kill -TERM -- -$JOB 2>/dev/null
  for _ in $(seq "$STOP_GRACE_S"); do kill -0 $JOB 2>/dev/null || break; sleep 1; done
  if kill -0 $JOB 2>/dev/null; then
    say "still running after ${STOP_GRACE_S} s: SIGKILL"
    kill -KILL -- -$JOB 2>/dev/null
  fi
  wait $JOB 2>/dev/null
  exit 2
}
trap 'stop "guard received a stop signal (job cancelled or node stopping)"' TERM INT HUP

say "job pid $JOB started on $(hostname -s): free<${HARD_MIN_FREE}% or <${MIN_FREE}% for ${LOW_MEM_CHECKS} checks" \
    "$([ "$GUARD_DNS" = 1 ] && echo ", dns fail x${DNS_FAIL_LIMIT} or >${DNS_MAX_MS}ms x3, AdGuard gone")" \
    "(pre: free ${pre_free}%, dns ${pre_dns} ms; MPS cap ${PYTORCH_MPS_HIGH_WATERMARK_RATIO}, ${THREADS} threads, qos_bg=${QOS_BG})"
dns_fail=0 dns_slow=0 mem_low=0 n=0
while kill -0 $JOB 2>/dev/null; do
  free=$(free_pct)
  if [ -n "$free" ] && [ "$free" -lt "$MIN_FREE" ]; then mem_low=$((mem_low + 1)); else mem_low=0; fi
  [ -n "$free" ] && [ "$free" -lt "$HARD_MIN_FREE" ] && stop "free memory ${free}% < ${HARD_MIN_FREE}%"
  [ "$mem_low" -ge "$LOW_MEM_CHECKS" ] && stop "free memory below ${MIN_FREE}% for $mem_low checks"
  [ -n "$WATCH_PID" ] && ! kill -0 "$WATCH_PID" 2>/dev/null && stop "queue worker (pid $WATCH_PID) is gone"
  dns=-
  if [ "$GUARD_DNS" = 1 ]; then
    dns=$(dns_ms)
    if [ -z "$dns" ]; then dns_fail=$((dns_fail + 1)); dns_slow=0
    elif [ "$dns" -gt "$DNS_MAX_MS" ]; then dns_slow=$((dns_slow + 1)); dns_fail=0
    else dns_fail=0 dns_slow=0; fi
    pgrep -q AdGuardHome || stop "AdGuardHome process not running"
    [ "$dns_fail" -ge "$DNS_FAIL_LIMIT" ] && stop "DNS failed $dns_fail checks in a row"
    [ "$dns_slow" -ge 3 ] && stop "DNS slower than ${DNS_MAX_MS} ms $dns_slow checks in a row"
  fi
  # A status line every 5 minutes keeps the job log readable.
  [ $((n % 30)) = 0 ] && say "free ${free:-?}% | dns ${dns:-FAIL} ms"
  n=$((n + 1))
  sleep 10 & wait $!   # wait (not sleep) so a stop signal is handled at once
done
wait $JOB; rc=$?
say "job exited with code $rc"
exit $rc
