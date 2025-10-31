# build_index/__init__.py
"""
build_index — офлайн-конвейер подготовки данных:
CSV→Parquet, теги/алиасы/импликации, биты/меммапы, статистика, PMI, Top-K.
"""
from .config import Config
from .io_stage import step_parquet
from .tags_stage import step_tags_and_post_tags, step_implications
from .index_stage import step_build_bitmaps, step_build_mmaps
from .stats_stage import step_tag_stats, step_pmi, step_topk
from .tag2vec_stage import step_tag2vec

__all__ = [
    "Config",
    "step_parquet",
    "step_tags_and_post_tags",
    "step_implications",
    "step_build_bitmaps",
    "step_build_mmaps",
    "step_tag_stats",
    "step_pmi",
    "step_topk",
    "step_tag2vec"
]