@echo off
title MediPredict AI Setup and Launcher
echo ==========================================
echo    MediPredict AI Setup and Launcher
echo ==========================================
echo.

:: 1. Check if Python is installed
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not installed or not added to PATH.
    echo Attempting to install Python using winget...
    winget install -e --id Python.Python.3.12 --source winget --accept-package-agreements --accept-source-agreements
    
    echo.
    echo =======================================================
    echo Python installation triggered. 
    echo Please CLOSE this window and DOUBLE-CLICK this script
    echo again to restart with Python enabled in your PATH.
    echo =======================================================
    pause
    exit /b
)

:: 2. Set up a Python Virtual Environment
echo [1/4] Setting up Python Virtual Environment...
IF NOT EXIST "venv" (
    python -m venv venv
    echo Virtual environment created.
)
call venv\Scripts\activate.bat

:: 3. Install required packages
echo.
echo [2/4] Checking and installing dependencies...
echo Updating pip...
python -m pip install --upgrade pip
echo.
echo Installing packages from requirements.txt...
pip install -r requirements.txt

:: 4. Generate data and train model if necessary
echo.
echo [3/4] Checking if model is trained...
IF NOT EXIST "model\best_model.pkl" (
    echo Model not found. Running first-time setup...
    echo Generating datasets...
    python data\generate_dataset.py
    echo Training machine learning model...
    python model\train.py
) ELSE (
    echo Model is already trained and ready!
)

:: 5. Run the web application
echo.
echo [4/4] Starting MediPredict AI Web Server...
echo Opening your default web browser...

:: Wait a brief moment to ensure server has time to start, then open browser
start http://127.0.0.1:5000

python app.py

pause
