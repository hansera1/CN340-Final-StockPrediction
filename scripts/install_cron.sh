#!/usr/bin/env bash
# Task 9 / VM: install the cron jobs of this project for the current user (run as user "forecast", not root).
# Equivalent of scripts/setup_tasks.ps1. Times are read from configs/rules.yaml.
#
#   bash scripts/install_cron.sh                 # evening job 17:30 + watchdogs (deadline - 10 min)
#   bash scripts/install_cron.sh --with-predict  # + 05_predict_live.py at the cutoffs (after the cutoff is measured)
#   bash scripts/install_cron.sh --remove        # remove this project's block
#
# - Monday-Friday; market holidays in data/external/calendar.csv are skipped by the scripts (is_trading_day)
# - CRON_TZ=Asia/Bangkok (the system timezone is also Asia/Bangkok via setup_vm.sh; scripts stop if it is not UTC+7)
# - flock per job (run_task.sh) · logs in outputs/logs/cron_<job>.log
# - re-run after changing cutoff / submit_deadline in rules.yaml
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
MODE="${1:-}"
BEGIN="# BEGIN kbank_advanc_forecast"
END="# END kbank_advanc_forecast"

current="$(crontab -l 2>/dev/null || true)"
others="$(printf '%s\n' "$current" | sed "/^${BEGIN}\$/,/^${END}\$/d")"

if [ "$MODE" = "--remove" ]; then
    printf '%s\n' "$others" | crontab -
    echo "removed the kbank_advanc_forecast block"; exit 0
fi

read -r AM_M AM_H PM_M PM_H WAM_M WAM_H WPM_M WPM_H < <(.venv/bin/python - "$ROOT/configs/rules.yaml" <<'EOF'
import sys, yaml, datetime as d
r = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))["sessions"]
def hm(s, k=0):
    t = d.datetime.strptime(s, "%H:%M") - d.timedelta(minutes=k)
    return f"{t.minute} {t.hour}"
print(hm(r["morning"]["cutoff"]), hm(r["afternoon"]["cutoff"]),
      hm(r["morning"]["submit_deadline"], 10), hm(r["afternoon"]["submit_deadline"], 10))
EOF
)

RUN="$ROOT/scripts/run_task.sh"
block="$BEGIN
CRON_TZ=Asia/Bangkok
30 17 * * 1-5 $RUN collect_daily 00_collect_daily.py
$WAM_M $WAM_H * * 1-5 $RUN watchdog_morning 10_watchdog.py morning
$WPM_M $WPM_H * * 1-5 $RUN watchdog_afternoon 10_watchdog.py afternoon"
if [ "$MODE" = "--with-predict" ]; then
    block="$block
$AM_M $AM_H * * 1-5 $RUN predict_morning 05_predict_live.py morning
$PM_M $PM_H * * 1-5 $RUN predict_afternoon 05_predict_live.py afternoon"
fi
block="$block
$END"

chmod +x "$RUN"
printf '%s\n%s\n' "$others" "$block" | sed '/^$/N;/^\n$/D' | crontab -
echo "installed:"; crontab -l | sed -n "/^${BEGIN}\$/,/^${END}\$/p"
echo "logs: $ROOT/outputs/logs/cron_<job>.log"
