#!/bin/bash
# Run a command under a memory guard, and kill the command — never the host.
#
#   ./runguard.sh -- python run_pipeline.py ...
#   ./runguard.sh --max-rss 8 --min-avail 20 -- python train_chunked.py ...
#
# Why this exists
# ---------------
# This box is an LXC guest with no cgroup memory cap and no swap, so a process
# that overcommits takes the *host* down with it — not just itself. That
# happened on 2026-09-16: run_pipeline.py's split materialised its arrays and
# MemAvailable fell from 15.43 GiB to 4.39 GiB in six seconds, and the machine
# rebooted.
#
# ram_watchdog.sh guards the poly_data collection pipeline only. Anything run
# by hand — a training run, a sweep, a conversion — is unguarded unless it goes
# through here.
#
# Two independent limits, because they fail differently:
#   * --max-rss    caps THIS command. Catches runaway growth early, before it
#                  is everyone else's problem.
#   * --min-avail  watches the whole machine. Catches the case where this
#                  command is innocent but something else is eating memory.
#
# The poll interval is deliberately short. The 2026-09-16 crash consumed 11 GiB
# in six seconds; a five-second poll would have missed it.

MAX_RSS_GB="${MAX_RSS_GB:-12}"      # kill the command above this RSS
MIN_AVAIL_GB="${MIN_AVAIL_GB:-16}"  # kill it if the machine drops below this
INTERVAL="${INTERVAL:-2}"           # seconds between checks
LOG="${RUNGUARD_LOG:-/root/Neuropoly/runguard.log}"

usage() { sed -n '2,30p' "$0"; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --max-rss)    MAX_RSS_GB="$2"; shift 2 ;;
    --min-avail)  MIN_AVAIL_GB="$2"; shift 2 ;;
    --interval)   INTERVAL="$2"; shift 2 ;;
    -h|--help)    usage 0 ;;
    --)           shift; break ;;
    *)            echo "unknown option: $1" >&2; usage 2 ;;
  esac
done
[ $# -eq 0 ] && { echo "nothing to run" >&2; usage 2; }

ts()  { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

avail_gb() { awk '/^MemAvailable:/{printf "%.2f", $2/1048576}' /proc/meminfo; }
tree_rss_gb() {   # RSS of the process and every descendant
  local root="$1" pids="$1" next
  while :; do
    next="$(pgrep -P "$(echo "$pids" | tr '\n' ',' | sed 's/,$//')" 2>/dev/null)"
    [ -z "$next" ] && break
    case "$pids" in *"$next"*) break ;; esac
    pids="$pids
$next"
  done
  ps -o rss= -p "$(echo "$pids" | tr '\n' ',' | sed 's/,$//')" 2>/dev/null \
    | awk '{s+=$1} END{printf "%.2f", s/1048576}'
}

kill_tree() {
  local pid="$1" sig="${2:-TERM}" kid
  for kid in $(pgrep -P "$pid" 2>/dev/null); do kill_tree "$kid" "$sig"; done
  kill -"$sig" "$pid" 2>/dev/null
}

log "=================================================="
log "guarding: $*"
log "limits: max-rss ${MAX_RSS_GB}G, min-avail ${MIN_AVAIL_GB}G, poll ${INTERVAL}s"

"$@" &
CHILD=$!
KILLED=""
PEAK=0

while kill -0 "$CHILD" 2>/dev/null; do
  rss="$(tree_rss_gb "$CHILD")"; rss="${rss:-0}"
  avail="$(avail_gb)"
  awk -v a="$rss" -v b="$PEAK" 'BEGIN{exit !(a>b)}' && PEAK="$rss"

  if awk -v r="$rss" -v m="$MAX_RSS_GB" 'BEGIN{exit !(r>m)}'; then
    KILLED="RSS ${rss}G exceeded --max-rss ${MAX_RSS_GB}G"
  elif awk -v a="$avail" -v m="$MIN_AVAIL_GB" 'BEGIN{exit !(a<m)}'; then
    KILLED="MemAvailable ${avail}G below --min-avail ${MIN_AVAIL_GB}G (job RSS ${rss}G)"
  fi

  if [ -n "$KILLED" ]; then
    log "LIMIT HIT: $KILLED"
    { free -h; ps -eo pid,ppid,%mem,rss,cmd --sort=-%mem | head -12; } >> "$LOG" 2>&1
    kill_tree "$CHILD" TERM
    for _ in $(seq 1 10); do kill -0 "$CHILD" 2>/dev/null || break; sleep 1; done
    kill -0 "$CHILD" 2>/dev/null && kill_tree "$CHILD" KILL
    wait "$CHILD" 2>/dev/null
    log "killed. peak RSS was ${PEAK}G"
    exit 137
  fi
  sleep "$INTERVAL"
done

wait "$CHILD"; rc=$?
log "finished rc=$rc, peak RSS ${PEAK}G"
exit "$rc"
