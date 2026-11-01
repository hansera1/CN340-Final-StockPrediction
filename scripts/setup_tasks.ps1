# Task 9 item 6: register Windows Task Scheduler jobs (same file for the main and the backup machine).
# Times are read from configs/rules.yaml. ASCII only: Windows PowerShell 5.1 misreads UTF-8 without BOM.
#
#   powershell -ExecutionPolicy Bypass -File scripts\setup_tasks.ps1                # evening job + watchdogs
#   powershell -ExecutionPolicy Bypass -File scripts\setup_tasks.ps1 -WithPredict   # + run 05_predict_live.py at the cutoffs
#   powershell -ExecutionPolicy Bypass -File scripts\setup_tasks.ps1 -Remove        # remove every job
#
# - Monday-Friday only; market holidays in data/external/calendar.csv are skipped by the scripts (is_trading_day)
# - evening job / predictions: "Wake the computer to run this task" + "Run task as soon as possible after a missed start"
# - watchdogs: wake only (a late run after the deadline would raise a false alarm)
# - re-run this script after changing cutoff / submit_deadline in rules.yaml
param([switch]$WithPredict, [switch]$Remove)

$root = Split-Path -Parent $PSScriptRoot
$wrapper = Join-Path $PSScriptRoot "run_task.cmd"
$names = "kbank_advanc_collect_daily", "kbank_advanc_watchdog_morning", "kbank_advanc_watchdog_afternoon",
         "kbank_advanc_predict_morning", "kbank_advanc_predict_afternoon"

foreach ($n in $names) { Unregister-ScheduledTask -TaskName $n -Confirm:$false -ErrorAction SilentlyContinue }
if ($Remove) { "removed all kbank_advanc_* tasks"; return }

$rulesPath = Join-Path $root "configs\rules.yaml"
$py = "import sys, yaml, datetime as d; r = yaml.safe_load(open(sys.argv[1], encoding='utf-8'))['sessions']; " +
      "m = lambda s, k=0: (d.datetime.strptime(s, '%H:%M') - d.timedelta(minutes=k)).strftime('%H:%M'); " +
      "print(m(r['morning']['cutoff']), m(r['afternoon']['cutoff']), " +
      "m(r['morning']['submit_deadline'], 10), m(r['afternoon']['submit_deadline'], 10))"
$t = (python -c $py $rulesPath).Trim().Split(" ")
$days = "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"

function Add-Task($name, $time, $scriptArgs, [switch]$CatchUp) {
    $action = New-ScheduledTaskAction -Execute $wrapper -Argument $scriptArgs -WorkingDirectory $root
    $trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $days -At $time
    $set = New-ScheduledTaskSettingsSet -WakeToRun -StartWhenAvailable:$CatchUp -AllowStartIfOnBatteries `
           -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Hours 1)
    Register-ScheduledTask -TaskName $name -Action $action -Trigger $trigger -Settings $set -Force | Out-Null
    $mode = if ($CatchUp) { "wake + run if missed" } else { "wake" }
    "{0,-34} {1}  {2}  ({3})" -f $name, $time, $scriptArgs, $mode
}

Add-Task "kbank_advanc_collect_daily" "17:30" "00_collect_daily.py" -CatchUp
Add-Task "kbank_advanc_watchdog_morning" $t[2] "10_watchdog.py morning"
Add-Task "kbank_advanc_watchdog_afternoon" $t[3] "10_watchdog.py afternoon"
if ($WithPredict) {
    Add-Task "kbank_advanc_predict_morning" $t[0] "05_predict_live.py morning" -CatchUp
    Add-Task "kbank_advanc_predict_afternoon" $t[1] "05_predict_live.py afternoon" -CatchUp
}
"logs: outputs\logs\task_<script>.log"
