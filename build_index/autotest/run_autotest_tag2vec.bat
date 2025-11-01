@echo off
call conda init
call conda activate myenv
python "I:\TIRESIAS_ENGINE\autotest\autotest_tag2vec.py" --root-data "I:\TIRESIAS_ENGINE\data" --tag car --topk 10
pause