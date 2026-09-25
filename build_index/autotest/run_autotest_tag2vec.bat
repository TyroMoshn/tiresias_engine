@echo off
setlocal ENABLEDELAYEDEXPANSION
REM TIRESIAS_ENGINE / Autotest Tag2Vec launcher

set TESTS_DIR=%~dp0
if "%TIRESIAS_ROOT%"=="" set TIRESIAS_ROOT=%TESTS_DIR%..\..
if "%TIRESIAS_DATA_ROOT%"=="" set TIRESIAS_DATA_ROOT=%TIRESIAS_ROOT%\data

set PYTHONUTF8=1
set PYTHONPATH=%TIRESIAS_ROOT%;%PYTHONPATH%

call conda activate myenv 2>nul

python "%TESTS_DIR%autotest_tag2vec.py" --root-data "%TIRESIAS_DATA_ROOT%" --tag car --topk 10
pause