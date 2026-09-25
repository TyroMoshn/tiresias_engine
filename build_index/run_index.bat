@echo off
setlocal ENABLEDELAYEDEXPANSION

REM TIRESIAS_ENGINE - Offline Indexing Pipeline Runner
set SCRIPT_DIR=%~dp0
if "%TIRESIAS_ROOT%"=="" set TIRESIAS_ROOT=%SCRIPT_DIR%..
if "%TIRESIAS_DATA_ROOT%"=="" set TIRESIAS_DATA_ROOT=%TIRESIAS_ROOT%\data

set PYTHONUTF8=1
set PYTHONPATH=%TIRESIAS_ROOT%;%PYTHONPATH%

call conda activate myenv 2>nul

cd /d "%TIRESIAS_ROOT%"
python -m build_index.main ^
	--root "%TIRESIAS_DATA_ROOT%" ^
	--do all ^
	--workers 8 ^
	--pool-min-size 3 ^
	--pool-max-size 200 ^
	--pools-collection-entropy-max 5.5 ^
	--pmi-support 50 ^
	--pmi-top-m-per-post 16 ^
	--tag2vec-dim 128 ^
	--tag2vec-min-df 200 ^
	--tag2vec-source merge ^
	--tag2vec-pool-alpha 0.5 ^
	--tag2vec-shift 0.0 ^
	--tag2vec-knn-k 100 ^
	--taste-archetypes-k 64 ^
	--taste-archetypes-top-posts 500 ^
	--cofav-min-weight 0.5 ^
	--post2vec-sq8
pause
