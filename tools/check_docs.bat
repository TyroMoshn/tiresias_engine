@echo off
setlocal ENABLEDELAYEDEXPANSION

REM TIRESIAS ENGINE - Documentation Freshness Checker (Windows)
set SCRIPT_DIR=%~dp0
if "%TIRESIAS_ROOT%"=="" set TIRESIAS_ROOT=%SCRIPT_DIR%..

set PYTHONUTF8=1
set PYTHONPATH=%TIRESIAS_ROOT%;%PYTHONPATH%

REM 1. Explicit override
if not "%PYTHON_EXE%"=="" goto :run

REM 2. Active Virtualenv or Conda
if not "%VIRTUAL_ENV%"=="" if exist "%VIRTUAL_ENV%\Scripts\python.exe" (
    set "PYTHON_EXE=%VIRTUAL_ENV%\Scripts\python.exe"
    goto :run
)
if not "%CONDA_PREFIX%"=="" if exist "%CONDA_PREFIX%\python.exe" (
    set "PYTHON_EXE=%CONDA_PREFIX%\python.exe"
    goto :run
)

REM 3. Local virtual environments
if exist "%TIRESIAS_ROOT%\venv\Scripts\python.exe" (
    set "PYTHON_EXE=%TIRESIAS_ROOT%\venv\Scripts\python.exe"
    goto :run
)
if exist "%TIRESIAS_ROOT%\.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%TIRESIAS_ROOT%\.venv\Scripts\python.exe"
    goto :run
)

REM 4. Standard Python launcher (py)
where py >nul 2>&1
if not errorlevel 1 (
    for /f "tokens=*" %%I in ('py -3 -c "import sys; print(sys.executable)" 2^>nul') do (
        set "PYTHON_EXE=%%I"
        goto :run
    )
)

REM 5. System PATH python
where python >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_EXE=python"
    goto :run
)

REM 6. Common Windows locations
if exist "C:\Python312\python.exe" set "PYTHON_EXE=C:\Python312\python.exe" & goto :run
if exist "C:\Python311\python.exe" set "PYTHON_EXE=C:\Python311\python.exe" & goto :run

set "PYTHON_EXE=python"

:run
cd /d "%TIRESIAS_ROOT%"
"%PYTHON_EXE%" -m tools.check_docs %*
exit /b %ERRORLEVEL%
