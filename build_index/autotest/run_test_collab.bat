@echo off
setlocal ENABLEDELAYEDEXPANSION

REM TIRESIAS_ENGINE - Run Collab Unit Tests
set SCRIPT_DIR=%~dp0
if "%TIRESIAS_ROOT%"=="" set TIRESIAS_ROOT=%SCRIPT_DIR%..\..
if "%TIRESIAS_DATA_ROOT%"=="" set TIRESIAS_DATA_ROOT=%TIRESIAS_ROOT%\data

set PYTHONUTF8=1
set PYTHONPATH=%TIRESIAS_ROOT%;%PYTHONPATH%

call conda activate myenv 2>nul

python "%SCRIPT_DIR%test_collab.py"
if %ERRORLEVEL% equ 0 (
    echo Collab Tests: OK
) else (
    echo [ERROR] Collab Tests Failed!
)
pause
