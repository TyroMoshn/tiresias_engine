@echo off
call conda init
call conda activate myenv
cd /d I:\TIRESIAS_ENGINE
python -m build_index.main ^
	--root I:\TIRESIAS_ENGINE\data ^
	--do tags post_tags parquet stats pmi pools mmaps bitmaps topk pools_entropy pools_tag_co tag2vec post2vec ^
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
	--tag2vec-knn-k 100
pause
