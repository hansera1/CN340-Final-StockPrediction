@echo off
rem Task Scheduler wrapper: run_task.cmd <script.py> [args...]  ->  outputs\logs\task_<script>.log
rem uses "python" on PATH so the same file works on the main and the backup machine (scripts\setup_tasks.ps1)
cd /d "%~dp0.."
if not exist outputs\logs mkdir outputs\logs
python scripts\%* >> "outputs\logs\task_%~n1.log" 2>&1
