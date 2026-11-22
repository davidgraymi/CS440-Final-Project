#!/usr/bin/env bash
# Start or stop this Mac's node of the two-Mac Ray cluster behind c4-queue.
#
#   scripts/ray_node.sh head              # on the Mac mini (always on): starts the cluster
#   scripts/ray_node.sh join <head-host>  # on the MacBook, e.g. join macmini.local
#   scripts/ray_node.sh status            # nodes, free job slots, queued work
#   scripts/ray_node.sh stop              # leave the cluster; stops this Mac's running jobs
#
# Each Mac offers SLOTS job slots (the custom Ray resource "mps"); a job takes
# one slot by default, so a Mac runs at most SLOTS jobs at once and further
# jobs wait in the queue until a slot frees up on either Mac.
#
# Both Macs need: this repo at the same commit, `uv sync` done (same Python and
# Ray version), and the same Ray auth token in ~/.ray/auth_token. The head
# creates the token; copy that file to the MacBook once (AirDrop is fine) or
# export RAY_AUTH_TOKEN=<its contents> before `join`.
#
# On macOS, Ray binds to 127.0.0.1 unless given an address: start the head with
# NODE_IP=<its LAN address> (e.g. NODE_IP=192.168.0.253 scripts/ray_node.sh head).
# The macOS firewall must allow incoming connections for gcs_server, raylet and python.
# Overrides: SLOTS=1 RAY_PORT=6379 NUM_CPUS=2 OBJ_STORE_MB=300 NODE_IP=<auto>
#   C4_GUARD_DNS (1 on head, 0 on join: whether jobs here also guard AdGuard DNS)
set -euo pipefail
cd "$(dirname "$0")/.."

RAY=.venv/bin/ray
[ -x "$RAY" ] || { echo "no $RAY: run 'uv sync' in $(pwd) first"; exit 1; }
SLOTS=${SLOTS:-1}
PORT=${RAY_PORT:-6379}

export RAY_AUTH_MODE=token          # nobody without the token can run code on the cluster
export RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1  # Ray refuses multi-node clusters on macOS without this
export RAY_USAGE_STATS_ENABLED=0
export C4_REPO=$(pwd)               # job processes inherit this: where to run and write results

common=(--num-cpus "${NUM_CPUS:-2}" --include-dashboard=false --disable-usage-stats
        --object-store-memory $(( ${OBJ_STORE_MB:-300} * 1000000 )))
[ -n "${NODE_IP:-}" ] && common+=(--node-ip-address "$NODE_IP")

need_token() {
  if [ -z "${RAY_AUTH_TOKEN:-}" ] && [ ! -s "$HOME/.ray/auth_token" ]; then
    echo "no Ray auth token: copy ~/.ray/auth_token from the head Mac to this one"
    echo "(or export RAY_AUTH_TOKEN=<its contents>), then run this again"
    exit 1
  fi
}

case "${1:-}" in
  head)
    export C4_GUARD_DNS=${C4_GUARD_DNS:-1}
    [ -s "$HOME/.ray/auth_token" ] || "$RAY" get-auth-token --generate >/dev/null
    nice -n 10 "$RAY" start --head --port "$PORT" "${common[@]}" \
      --resources "{\"mps\": $SLOTS, \"c4_head\": 1}"
    echo
    echo "head up with $SLOTS job slot(s). On the MacBook: scripts/ray_node.sh join $(hostname -s).local"
    ;;
  join)
    [ -n "${2:-}" ] || { echo "usage: $0 join <head-host>"; exit 64; }
    need_token
    export C4_GUARD_DNS=${C4_GUARD_DNS:-0}
    nice -n 10 "$RAY" start --address "$2:$PORT" "${common[@]}" --resources "{\"mps\": $SLOTS}"
    echo
    echo "joined $2 with $SLOTS job slot(s). Keep this Mac awake and on power while it has jobs."
    ;;
  status) "$RAY" status ;;
  stop)   "$RAY" stop ;;
  *) sed -n '2,7p' "$0" | sed 's/^# \{0,1\}//'; exit 64 ;;
esac
