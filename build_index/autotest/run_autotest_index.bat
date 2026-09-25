@echo off
setlocal ENABLEDELAYEDEXPANSION
REM TIRESIAS_ENGINE / Autotest launcher

set TESTS_DIR=%~dp0
if "%TIRESIAS_ROOT%"=="" set TIRESIAS_ROOT=%TESTS_DIR%..\..
if "%TIRESIAS_DATA_ROOT%"=="" set TIRESIAS_DATA_ROOT=%TIRESIAS_ROOT%\data

set PYTHONUTF8=1
set PYTHONPATH=%TIRESIAS_ROOT%;%PYTHONPATH%

call conda activate myenv 2>nul

python "%TESTS_DIR%autotest_index.py" --root "%TIRESIAS_DATA_ROOT%" --report "%TESTS_DIR%autotest_report.txt" --manifest "%TESTS_DIR%autotest_manifest.json" --sample-tags 10 --k-cap 1000 --emit-bat 0
set ERR=%ERRORLEVEL%
if %ERR% NEQ 0 (
  echo Autotest Index: FAIL (exit code %ERR%)
) else (
  echo Autotest Index: OK
)
pause