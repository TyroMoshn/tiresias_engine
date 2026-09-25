@echo off
setlocal ENABLEDELAYEDEXPANSION

REM TIRESIAS_ENGINE - Recommendation Serving Server Launcher
set SCRIPT_DIR=%~dp0
if "%TIRESIAS_ROOT%"=="" set TIRESIAS_ROOT=%SCRIPT_DIR%..
if "%TIRESIAS_DATA_ROOT%"=="" set TIRESIAS_DATA_ROOT=%TIRESIAS_ROOT%\data

set PYTHONUTF8=1
set PYTHONPATH=%SCRIPT_DIR%;%TIRESIAS_ROOT%;%PYTHONPATH%

call conda activate myenv 2>nul

echo Checking server dependencies...
python -c "import fastapi, uvicorn, pydantic, psutil" 2>nul
if %ERRORLEVEL% neq 0 (
    echo Installing required server packages: fastapi, uvicorn, pydantic, psutil ...
    call python -m pip install -r "%SCRIPT_DIR%requirements.txt"
)

echo.
echo Starting TIRESIAS_ENGINE Serving Server...
echo API documentation will be available at: http://127.0.0.1:8000/docs
echo.

REM In the standard Windows cmd.exe window, color display is disabled by default, so colors are printed as raw special characters.
REM The launch command, which clutters the log a bit:
REM python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --no-use-colors

pause
