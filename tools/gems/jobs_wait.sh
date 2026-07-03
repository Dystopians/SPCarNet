#!/bin/bash
# GEMS canonical job waiter: blocks until every named job reaches a terminal
# state, then prints one line per job and exits 0 (any failure -> exit 1).
# Terminal states: EXITED(code) via <name>.exit; KILLED via dead PID with no
# exit file. No pgrep, no pattern matching, no log heuristics.
# Usage: jobs_wait.sh <job-name> [<job-name>...]
set -u
JOBS=${GEMS_JOBS_DIR:-/data/peilincai/gems_stage1/jobs}
state() { # name -> RUNNING | EXITED:<code> | KILLED | UNKNOWN
  local n=$1 pidf="$JOBS/$1.pid" exitf="$JOBS/$1.exit"
  if [ -f "$exitf" ]; then echo "EXITED:$(cat "$exitf")"; return; fi
  if [ -f "$pidf" ] && kill -0 "$(cat "$pidf")" 2>/dev/null; then echo RUNNING; return; fi
  if [ -f "$pidf" ]; then echo KILLED; return; fi
  echo UNKNOWN
}
while :; do
  alldone=1
  for n in "$@"; do
    s=$(state "$n"); [ "$s" = RUNNING ] && alldone=0
  done
  [ $alldone -eq 1 ] && break
  sleep 30
done
fail=0
for n in "$@"; do
  s=$(state "$n"); echo "$n: $s"
  case "$s" in EXITED:0) ;; *) fail=1;; esac
done
exit $fail
