@echo off 
:: Navigate to the main AUTO-TRIAGE folder (one level up from this bat file)
cd /d "%~dp0\.."

echo Running Auto_TRIAGE manual prediction demo...
echo.

:: Use the virtual environment to run the script inside working_models
venv\Scripts\python.exe working_models\runxgb.py

echo.
pause