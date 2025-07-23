@echo off
title Parking Management System
cd /d "%~dp0"

echo Starting Parking Management System...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

REM Check if required packages are installed
echo Checking requirements...
python -c "import flask, cv2, ultralytics, werkzeug" >nul 2>&1
if errorlevel 1 (
    echo Installing required packages...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo Failed to install requirements
        pause
        exit /b 1
    )
)

echo.
echo Starting application...
python run_app.py

pause
