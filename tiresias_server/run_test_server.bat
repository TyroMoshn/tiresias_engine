@echo off
setlocal ENABLEDELAYEDEXPANSION

REM TIRESIAS_ENGINE - Run Serving Core Unit Tests
set SCRIPT_DIR=%~dp0
if "%TIRESIAS_ROOT%"=="" set TIRESIAS_ROOT=%SCRIPT_DIR%..
if "%TIRESIAS_DATA_ROOT%"=="" set TIRESIAS_DATA_ROOT=%TIRESIAS_ROOT%\data

set PYTHONUTF8=1
set PYTHONPATH=%SCRIPT_DIR%;%TIRESIAS_ROOT%;%PYTHONPATH%

call conda activate myenv 2>nul

echo Checking test dependencies...
python -c "import fastapi, uvicorn, pydantic, psutil" 2>nul
if %ERRORLEVEL% neq 0 (
    echo Installing server test packages: fastapi, uvicorn, pydantic, psutil ...
    call python -m pip install -r "%SCRIPT_DIR%requirements.txt"
)

echo.
echo Running Serving Core Unit Tests...
echo.

python "%SCRIPT_DIR%autotest\test_server.py"
if %ERRORLEVEL% equ 0 (
    echo.
    echo Serving Core Tests: OK
) else (
    echo.
    echo [ERROR] Serving Core Tests Failed!
)
pause
