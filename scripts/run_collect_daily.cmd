@echo off
rem Task Scheduler: kbank_advanc_collect_daily (Mon-Fri 17:30) - collect data, morning draft, E13 stage 2
cd /d "C:\Users\parin\Documents\kbank_advanc_forecast"
"C:\Users\parin\AppData\Local\Programs\Python\Python312\python.exe" scripts\00_collect_daily.py >> outputs\logs\collect_daily.log 2>&1

