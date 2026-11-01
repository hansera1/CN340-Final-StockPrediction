#!/usr/bin/env bash
# cron wrapper (VM): run_task.sh <job-name> <script.py> [args...]
#   - flock: the same job never runs twice at the same time (a second start exits and logs "skip")
#   - log: outputs/logs/cron_<job-name>.log  (inside the project, separate per machine)
#   - python from .venv (scripts/setup_vm.sh app)
set -u
cd "$(dirname "$0")/.." || exit 1
mkdir -p outputs/logs
job="$1"; shift
log="outputs/logs/cron_${job}.log"
exec 9>"/tmp/kbank_advanc_${job}.lock"
if ! flock -n 9; then
    echo "$(date '+%F %T %Z') ${job}: previous run still running -> skip" >> "$log"
    exit 0
fi
echo "===== $(date '+%F %T %Z') ${job}: $*" >> "$log"
.venv/bin/python "scripts/$1" "${@:2}" >> "$log" 2>&1
rc=$?
echo "===== exit ${rc}" >> "$log"
exit $rc
