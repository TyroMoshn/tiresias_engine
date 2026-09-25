@echo off
setlocal ENABLEDELAYEDEXPANSION
REM TIRESIAS_ENGINE / Visualize Post2Vec launcher

set TESTS_DIR=%~dp0
if "%TIRESIAS_ROOT%"=="" set TIRESIAS_ROOT=%TESTS_DIR%..\..
if "%TIRESIAS_DATA_ROOT%"=="" set TIRESIAS_DATA_ROOT=%TIRESIAS_ROOT%\data

set PYTHONUTF8=1
set PYTHONPATH=%TIRESIAS_ROOT%;%PYTHONPATH%

call conda activate myenv 2>nul

python "%TESTS_DIR%visualize_post2vec_cpu.py" ^
  --features-dir "%TIRESIAS_DATA_ROOT%\features" ^
  --data-root    "%TIRESIAS_DATA_ROOT%" ^
  --overlay-frac 0.02 ^
  --sample-size  2000000 ^
  --url-prefix "https://e621.net/posts/" ^
  --output "post2vec_umap_hdbscan.html"

pause