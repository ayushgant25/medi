@echo off
title Building MediPredict AI Executable
echo =======================================================
echo Preparing to build standalone executable (.exe)
echo =======================================================
echo.

echo [1/3] Activating Virtual Environment...
call venv\Scripts\activate.bat

echo.
echo [2/3] Installing dependencies and PyInstaller...
pip install -r requirements.txt
pip install pyinstaller

echo.
echo [3/3] Building the standalone .exe file...
echo This may take a few minutes as it bundles all dependencies...
pyinstaller --name "MediPredictAI" --onefile --add-data "templates;templates" --add-data "static;static" --add-data "model;model" --add-data "data;data" --hidden-import="sklearn.ensemble" --hidden-import="sklearn.tree" --hidden-import="sklearn.neighbors" --hidden-import="sklearn.naive_bayes" app.py

echo.
echo =======================================================
echo Build complete! 
echo You can find your standalone app here: dist\MediPredictAI.exe
echo =======================================================
