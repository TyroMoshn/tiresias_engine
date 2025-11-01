@echo off
call conda init
call conda activate myenv
setlocal ENABLEDELAYEDEXPANSION

set ROOT=I:\TIRESIAS_ENGINE
set DATA=%ROOT%\data
set BUILD=%ROOT%\build_index
set TESTS=%BUILD%\autotest

set PYTHONUTF8=1

python "%TESTS%\autotest_post2vec.py" --root-data "%DATA%" --sample 1 --tolerance 0.001 --query-post 3691518 --topk 10
set ERR=%ERRORLEVEL%
if %ERR% NEQ 0 (
  echo Autotest Post2Vec: FAIL (exit code %ERR%)
) else (
  echo Autotest Post2Vec: OK
)
pause