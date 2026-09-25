@echo off
setlocal ENABLEDELAYEDEXPANSION

REM TIRESIAS ENGINE - Automated Updater Launcher (Windows)
set SCRIPT_DIR=%~dp0
if "%TIRESIAS_ROOT%"=="" set TIRESIAS_ROOT=%SCRIPT_DIR%..
if "%TIRESIAS_DATA_ROOT%"=="" set TIRESIAS_DATA_ROOT=%TIRESIAS_ROOT%\data

set PYTHONUTF8=1
set PYTHONPATH=%TIRESIAS_ROOT%;%PYTHONPATH%

REM Locate Python interpreter dynamically without hardcoded paths or usernames:
REM 1. Caller explicit override
if not "%PYTHON_EXE%"=="" goto :found_python

REM ------------------------------------------------------------
REM PASS 1: Prioritize interpreters with FULL pipeline dependencies (duckdb + faiss)
REM ------------------------------------------------------------

REM 2. Active Virtualenv or Conda environment
if not "%VIRTUAL_ENV%"=="" if exist "%VIRTUAL_ENV%\Scripts\python.exe" (
    "%VIRTUAL_ENV%\Scripts\python.exe" -c "import duckdb, faiss" >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON_EXE=%VIRTUAL_ENV%\Scripts\python.exe"
        goto :found_python
    )
)
if not "%CONDA_PREFIX%"=="" if exist "%CONDA_PREFIX%\python.exe" (
    "%CONDA_PREFIX%\python.exe" -c "import duckdb, faiss" >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON_EXE=%CONDA_PREFIX%\python.exe"
        goto :found_python
    )
)

REM 3. Local virtual environments inside project directory (portable across any machine)
for %%V in ("%TIRESIAS_ROOT%\venv" "%TIRESIAS_ROOT%\.venv" "%TIRESIAS_ROOT%\env") do (
    if exist "%%~V\Scripts\python.exe" (
        "%%~V\Scripts\python.exe" -c "import duckdb, faiss" >nul 2>&1
        if not errorlevel 1 (
            set "PYTHON_EXE=%%~V\Scripts\python.exe"
            goto :found_python
        )
    )
)

REM 4. Dynamic scan of any Conda environments containing full pipeline dependencies (duckdb + faiss)
if exist "%USERPROFILE%\.conda\envs" (
    for /d %%E in (%USERPROFILE%\.conda\envs\*) do (
        if exist "%%E\python.exe" (
            "%%E\python.exe" -c "import duckdb, faiss" >nul 2>&1
            if not errorlevel 1 (
                set "PYTHON_EXE=%%E\python.exe"
                goto :found_python
            )
        )
    )
)
if exist "%USERPROFILE%\miniconda3\envs" (
    for /d %%E in (%USERPROFILE%\miniconda3\envs\*) do (
        if exist "%%E\python.exe" (
            "%%E\python.exe" -c "import duckdb, faiss" >nul 2>&1
            if not errorlevel 1 (
                set "PYTHON_EXE=%%E\python.exe"
                goto :found_python
            )
        )
    )
)
if exist "%USERPROFILE%\anaconda3\envs" (
    for /d %%E in (%USERPROFILE%\anaconda3\envs\*) do (
        if exist "%%E\python.exe" (
            "%%E\python.exe" -c "import duckdb, faiss" >nul 2>&1
            if not errorlevel 1 (
                set "PYTHON_EXE=%%E\python.exe"
                goto :found_python
            )
        )
    )
)
if exist "C:\ProgramData\anaconda3\envs" (
    for /d %%E in (C:\ProgramData\anaconda3\envs\*) do (
        if exist "%%E\python.exe" (
            "%%E\python.exe" -c "import duckdb, faiss" >nul 2>&1
            if not errorlevel 1 (
                set "PYTHON_EXE=%%E\python.exe"
                goto :found_python
            )
        )
    )
)

REM 5. Standard Python launcher (py) with full dependencies
where py >nul 2>&1
if not errorlevel 1 (
    for /f "tokens=*" %%I in ('py -3.12 -c "import duckdb, faiss, sys; print(sys.executable)" 2^>nul') do (
        set "PYTHON_EXE=%%I"
        goto :found_python
    )
    for /f "tokens=*" %%I in ('py -3.11 -c "import duckdb, faiss, sys; print(sys.executable)" 2^>nul') do (
        set "PYTHON_EXE=%%I"
        goto :found_python
    )
    for /f "tokens=*" %%I in ('py -3 -c "import duckdb, faiss, sys; print(sys.executable)" 2^>nul') do (
        set "PYTHON_EXE=%%I"
        goto :found_python
    )
)

REM 6. Standard system Python installations with full dependencies
for %%P in (
    "C:\Python312\python.exe"
    "C:\Python311\python.exe"
    "C:\Python310\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
) do (
    if exist %%P (
        %%P -c "import duckdb, faiss" >nul 2>&1
        if not errorlevel 1 (
            set "PYTHON_EXE=%%~P"
            goto :found_python
        )
    )
)

REM 7. System PATH python with full dependencies
where python >nul 2>&1
if not errorlevel 1 (
    for /f "tokens=*" %%I in ('python -c "import duckdb, faiss, sys; print(sys.executable)" 2^>nul') do (
        set "PYTHON_EXE=%%I"
        goto :found_python
    )
)

REM ------------------------------------------------------------
REM PASS 2: Fallback for environments with at least duckdb
REM ------------------------------------------------------------
if not "%VIRTUAL_ENV%"=="" if exist "%VIRTUAL_ENV%\Scripts\python.exe" (
    "%VIRTUAL_ENV%\Scripts\python.exe" -c "import duckdb" >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON_EXE=%VIRTUAL_ENV%\Scripts\python.exe"
        goto :found_python
    )
)
if not "%CONDA_PREFIX%"=="" if exist "%CONDA_PREFIX%\python.exe" (
    "%CONDA_PREFIX%\python.exe" -c "import duckdb" >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON_EXE=%CONDA_PREFIX%\python.exe"
        goto :found_python
    )
)
where py >nul 2>&1
if not errorlevel 1 (
    for /f "tokens=*" %%I in ('py -3.12 -c "import duckdb, sys; print(sys.executable)" 2^>nul') do (
        set "PYTHON_EXE=%%I"
        goto :found_python
    )
    for /f "tokens=*" %%I in ('py -3 -c "import duckdb, sys; print(sys.executable)" 2^>nul') do (
        set "PYTHON_EXE=%%I"
        goto :found_python
    )
)
for %%P in ("C:\Python312\python.exe" "C:\Python311\python.exe") do (
    if exist %%P (
        %%P -c "import duckdb" >nul 2>&1
        if not errorlevel 1 (
            set "PYTHON_EXE=%%~P"
            goto :found_python
        )
    )
)

REM ------------------------------------------------------------
REM FINAL FALLBACK: Default system Python
REM ------------------------------------------------------------
where py >nul 2>&1
if not errorlevel 1 (
    for /f "tokens=*" %%I in ('py -3 -c "import sys; print(sys.executable)" 2^>nul') do (
        set "PYTHON_EXE=%%I"
        goto :found_python
    )
)
set "PYTHON_EXE=python"

:found_python

cd /d "%TIRESIAS_ROOT%"

REM If arguments were passed from CLI, run directly
if not "%~1"=="" (
    echo [RUN] "%PYTHON_EXE%" -m tools.updater %*
    "%PYTHON_EXE%" -m tools.updater %*
    exit /b %ERRORLEVEL%
)

REM Interactive menu if launched without arguments (double-click)
echo ================================================================
echo           TIRESIAS ENGINE - AUTOMATED ARTIFACT UPDATER
echo ================================================================
echo Python:    !PYTHON_EXE!
echo Data Root: !TIRESIAS_DATA_ROOT!
echo.
echo Choose an action:
echo   [1] Full Update Cycle (--all: backup, uploaders, build, deploy)
echo   [2] Build Index Pipeline only (--step build --workers 12)
echo   [3] Extract Uploaders stats via DuckDB (--step uploaders)
echo   [4] Backup Database only (--step backup)
echo   [5] Deploy Artifacts to Targets (--step deploy)
echo   [6] Check Prerequisites (--step prereq)
echo   [0] Exit
echo ================================================================
set /p ACTION="Enter choice (default: 1): "
if "!ACTION!"=="" set ACTION=1

if "!ACTION!"=="1" (
    echo.
    echo Starting Full Update Cycle...
    "%PYTHON_EXE%" -m tools.updater --all --workers 12
) else if "!ACTION!"=="2" (
    echo.
    echo Starting Build Pipeline...
    "%PYTHON_EXE%" -m tools.updater --step build --workers 12
) else if "!ACTION!"=="3" (
    echo.
    echo Extracting Uploaders...
    "%PYTHON_EXE%" -m tools.updater --step uploaders --workers 8
) else if "!ACTION!"=="4" (
    echo.
    echo Backing up database...
    "%PYTHON_EXE%" -m tools.updater --step backup
) else if "!ACTION!"=="5" (
    echo.
    echo Deploying to configured targets...
    "%PYTHON_EXE%" -m tools.updater --step deploy
) else if "!ACTION!"=="6" (
    echo.
    echo Checking prerequisites...
    "%PYTHON_EXE%" -m tools.updater --step prereq
) else if "!ACTION!"=="0" (
    exit /b 0
) else (
    echo Invalid choice.
)

:done
echo.
echo [Finished with exit code %ERRORLEVEL%]
pause
