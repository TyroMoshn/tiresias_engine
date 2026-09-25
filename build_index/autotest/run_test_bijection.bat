@echo off
setlocal ENABLEDELAYEDEXPANSION
REM TIRESIAS_ENGINE / Bijection Unit Test Launcher

set TESTS_DIR=%~dp0
set PROJECT_ROOT=%TESTS_DIR%..\..
set PYTHONUTF8=1
set PYTHONPATH=%PROJECT_ROOT%;%PYTHONPATH%

call conda activate myenv 2>nul

python "%TESTS_DIR%test_bijection.py"
set ERR=%ERRORLEVEL%
if %ERR% NEQ 0 (
  echo Bijection Tests: FAIL (exit code %ERR%)
) else (
  echo Bijection Tests: OK
)
pause
