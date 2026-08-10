@echo off
rem Double-click to open the Lamellyx dashboard in your browser.
rem Starts a small local server and opens the page with an access token.
cd /d "%~dp0"
title Lamellyx

rem --- find a Python ------------------------------------------------------
set "PYEXE=python"
"%PYEXE%" --version >nul 2>&1
if not errorlevel 1 goto haspy

set "PYEXE=py"
"%PYEXE%" --version >nul 2>&1
if not errorlevel 1 goto haspy

echo.
echo   Python was not found.
echo.
echo   Install it from https://www.python.org/downloads/ and tick
echo   "Add python.exe to PATH" during setup, then run this file again.
echo.
pause
exit /b 1

:haspy
rem --- install the package if it is not there yet -------------------------
"%PYEXE%" -c "import lamellyx" >nul 2>&1
if not errorlevel 1 goto ready
echo Installing lamellyx (one time)...
"%PYEXE%" -m pip install --quiet -e .
if errorlevel 1 (
    echo.
    echo   Install failed. Try it by hand:  %PYEXE% -m pip install -e .
    echo.
    pause
    exit /b 1
)

:ready
echo.
echo   Starting the Lamellyx dashboard.
echo   A browser window will open in a moment.
echo.
echo   Keep this window open while you work - closing it stops the server.
echo.
"%PYEXE%" -m lamellyx dashboard %*
echo.
echo   Server stopped.
pause
