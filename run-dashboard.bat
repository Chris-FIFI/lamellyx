@echo off
title Lamellyx Dashboard
cd /d "%~dp0"
echo.
echo   Starting the Lamellyx dashboard...
echo   A browser tab will open automatically.
echo   Keep this window open while you use it; close it (or press Ctrl+C) to stop.
echo.
python -m lamellyx.dashboard
echo.
echo   The dashboard has stopped.
pause
