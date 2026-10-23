@echo off
title Kronos Forecast
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (
  py -3 launcher.py
) else (
  where python >nul 2>nul
  if errorlevel 1 (
    echo.
    echo Python was not found.
    echo Install Python 3.12 from https://www.python.org/downloads/
    echo and tick "Add python.exe to PATH" during setup, then double-click this file again.
    echo.
    pause
    exit /b 1
  )
  python launcher.py
)
echo.
pause
