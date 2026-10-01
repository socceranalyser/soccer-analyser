@echo off
REM Soccer Analyser: starts the dashboard and opens it in the browser.
REM Keep this window open while you use the dashboard; close it to stop the server.
cd /d "%~dp0"
title Soccer Analyser - dashboard
echo Starting Soccer Analyser... first load takes about 30 seconds.
echo Address: http://localhost:8501
start "" cmd /c "timeout /t 6 >nul & start http://localhost:8501"
".venv\Scripts\streamlit.exe" run app.py
pause
