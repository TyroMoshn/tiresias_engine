@echo off
setlocal ENABLEDELAYEDEXPANSION
REM TIRESIAS_ENGINE / Autotest Post2Vec launcher

set TESTS_DIR=%~dp0
if "%TIRESIAS_ROOT%"=="" set TIRESIAS_ROOT=%TESTS_DIR%..\..
if "%TIRESIAS_DATA_ROOT%"=="" set TIRESIAS_DATA_ROOT=%TIRESIAS_ROOT%\data

set PYTHONUTF8=1
set PYTHONPATH=%TIRESIAS_ROOT%;%PYTHONPATH%

call conda activate myenv 2>nul

python "%TESTS_DIR%autotest_post2vec.py" --root-data "%TIRESIAS_DATA_ROOT%" --sample 1 --tolerance 0.001 --query-post 3691518 --topk 10
set ERR=%ERRORLEVEL%
if %ERR% NEQ 0 (
  echo Autotest Post2Vec: FAIL (exit code %ERR%)
) else (
  echo Autotest Post2Vec: OK
)
pause