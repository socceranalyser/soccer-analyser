@echo off
REM Soccer Analyser: daily data refresh + forecast recording (run by Windows Task Scheduler).
REM Log: data\update.log
cd /d "%~dp0.."
echo ==== %date% %time% ==== >> data\update.log
".venv\Scripts\python.exe" scripts\update.py >> data\update.log 2>&1
echo exit code %errorlevel% >> data\update.log
