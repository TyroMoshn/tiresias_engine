@echo off
call conda init
call conda activate myenv
setlocal ENABLEDELAYEDEXPANSION
REM TIRESIAS_ENGINE / Autotest launcher

REM Root folder of the project (adjust if needed)
set ROOT=I:\\TIRESIAS_ENGINE
set DATA=%ROOT%\\data
set BUILD=%ROOT%\\build_index
set TESTS=%BUILD%\\autotest

REM Ensure UTF-8 in console
set PYTHONUTF8=1

REM Add build_index to PYTHONPATH to import Config
set PYTHONPATH=%BUILD%;%PYTHONPATH%

python "%TESTS%\\autotest_index.py" --root "%DATA%" --report "%TESTS%\\autotest_report.txt" --manifest "%TESTS%\\autotest_manifest.json" --sample-tags 10 --k-cap 1000 --emit-bat 0
set ERR=%ERRORLEVEL%
if %ERR% NEQ 0 (
  echo Autotest: FAIL (exit code %ERR%)
) else (
  echo Autotest: OK
)
pause