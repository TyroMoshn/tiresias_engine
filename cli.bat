@echo off
setlocal ENABLEDELAYEDEXPANSION

set SCRIPT_DIR=%~dp0
set SERVER_DIR=%SCRIPT_DIR%tiresias_server
set TIRESIAS_ROOT=%SCRIPT_DIR%
set TIRESIAS_DATA_ROOT=%TIRESIAS_ROOT%data

set PYTHONUTF8=1
set PYTHONPATH=%SERVER_DIR%;%TIRESIAS_ROOT%;%PYTHONPATH%

call conda activate myenv 2>nul

python -m app.cli %*
