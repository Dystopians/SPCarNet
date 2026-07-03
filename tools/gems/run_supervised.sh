#!/bin/bash
# GEMS canonical detached-job supervisor (LEDGER: zombie-proof protocol).
# Usage: run_supervised.sh <job-name> -- <command...>
# All state is FILES under $JOBS (no pattern-matching ever needed):
#   <name>.pid   written at launch (the setsid'd child's PID)
#   <name>.log   combined stdout+stderr
#   <name>.exit  written on completion with the exit code (missing + dead PID = external kill)
# Wait for jobs with jobs_wait.sh; inspect with jobs_status.sh.
set -u
JOBS=${GEMS_JOBS_DIR:-/data/peilincai/gems_stage1/jobs}
mkdir -p "$JOBS"
NAME=$1; shift
[ "$1" = "--" ] && shift
LOG="$JOBS/$NAME.log"; PIDF="$JOBS/$NAME.pid"; EXITF="$JOBS/$NAME.exit"
rm -f "$EXITF"
setsid nohup bash -c '
  echo $$ > '"$PIDF"'
  "$@" >> '"$LOG"' 2>&1
  echo $? > '"$EXITF"'
' _ "$@" < /dev/null > /dev/null 2>&1 &
disown
sleep 1
if [ -f "$PIDF" ]; then
  echo "launched $NAME pid=$(cat "$PIDF") log=$LOG"
else
  echo "LAUNCH FAILED: $NAME (no pidfile)"; exit 1
fi
