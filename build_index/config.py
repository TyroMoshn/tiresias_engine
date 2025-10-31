# build_index/config.py
from __future__ import annotations
import dataclasses, json, os
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    # корни и пути
    root: Path
    csv: Path
    posts_parquet: Path
    post_tags_parquet: Path
    bitmaps_dir: Path
    topk_dir: Path
    features_dir: Path
    tags_parquet: Path
    tags_csv: Path
    tag_aliases_csv: Path
    tag_implications_csv: Path
    # pools
    pools_csv: Path
    pools_parquet: Path
    pools_meta_parquet: Path
    pools_entropy_parquet: Path
    pools_edges_parquet: Path
    tag_co_from_pools_parquet: Path
    mmaps_dir: Path
    pool_min_size: int = 3
    pool_max_size: int = 200
    pools_use_series: bool = True
    pools_use_collections: bool = True
    pools_collection_entropy_max: float = 6.0
    pools_top_tags: int = 24
    tag_shards: int = 256        # сколько партиций хотим (кол-во файлов)
    parquet_row_group_rows: int = 1_000_000        # размер row group при записи (прибл.)
    # tag2vec
    tag2vec_dim: int = 128
    tag2vec_min_df: int = 100
    tag2vec_max_tags: int = 200_000
    tag2vec_source: str = "merge"       # 'merge' | 'pmi' | 'pools'
    tag2vec_pool_alpha: float = 0.5     # вес «пулов» при merge
    tag2vec_shift: float = 0.0          # сдвиг PMI (≈log(k) для нег. выборки)
    tag2vec_knn_k: int = 32

    # системное
    workers: int = os.cpu_count() or 4
    force: bool = False
    reliable_only: bool = True  # везде фильтровать deleted/pending

    # TOPK
    topk_k: int = 5000
    topk_mode: str = "static"         # 'static' | 'sqrt_df'
    topk_k_min: int = 100
    topk_k_max: int = 5000
    topk_beta: float = 10.0

    # IDF
    idf_alpha: float = 1_000.0
    idf_source: str = "auto"          # 'tags' | 'local' | 'auto'
    idf_auto_switch_threshold: float = 0.90

    # PMI/impl
    pmi_support: int = 50
    pmi_top_m_per_post: int = 16
    anc_cache_depth: int = 3
    anc_cache_top_percent: float = 0.05

    # roaring
    roar_shard_size: int = 0

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2, default=str)