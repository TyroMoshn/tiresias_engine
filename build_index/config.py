# build_index/config.py
from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict


def get_default_data_root() -> Path:
    """
    Resolve data root path portably across Windows and Linux.
    Prioritizes TIRESIAS_DATA_ROOT environment variable,
    falling back to 'data' directory adjacent to the project root.
    """
    env_root = os.environ.get("TIRESIAS_DATA_ROOT")
    if env_root:
        return Path(env_root).resolve()
    # __file__ is in <project_root>/build_index/config.py -> parents[1] is <project_root>
    return (Path(__file__).resolve().parents[1] / "data").resolve()


@dataclass
class Config:
    # paths and roots
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
    # uploaders
    uploaders_csv: Optional[Path] = None
    # collab & favorites
    user_favorites_csv: Optional[Path] = None
    post_cofav_parquet: Optional[Path] = None
    taste_archetypes_parquet: Optional[Path] = None
    taste_centroids_npy: Optional[Path] = None

    pool_min_size: int = 3
    pool_max_size: int = 200
    pools_use_series: bool = True
    pools_use_collections: bool = True
    pools_collection_entropy_max: float = 6.0
    pools_top_tags: int = 24
    tag_shards: int = 256  # How many partitions do we want (number of files)
    parquet_row_group_rows: int = 1_000_000  # row group size when written (approx.)

    # tag2vec
    tag2vec_dim: int = 128
    tag2vec_min_df: int = 100
    tag2vec_max_tags: int = 200_000
    tag2vec_source: str = "merge"  # 'merge' | 'pmi' | 'pools'
    tag2vec_pool_alpha: float = 0.5  # "pools" size when merge
    tag2vec_shift: float = 0.0  # PMI shift
    tag2vec_knn_k: int = 32

    # system
    workers: int = os.cpu_count() or 4
    force: bool = False
    reliable_only: bool = True  # filter everywhere deleted/pending

    # TOPK
    topk_k: int = 5000
    topk_mode: str = "static"  # 'static' | 'sqrt_df'
    topk_k_min: int = 100
    topk_k_max: int = 5000
    topk_beta: float = 10.0

    # IDF
    idf_alpha: float = 1_000.0
    idf_source: str = "auto"  # 'tags' | 'local' | 'auto'
    idf_auto_switch_threshold: float = 0.90

    # PMI/impl
    pmi_support: int = 50
    pmi_top_m_per_post: int = 16
    anc_cache_depth: int = 3
    anc_cache_top_percent: float = 0.05

    # roaring
    roar_shard_size: int = 0

    # post2vec v2 & categorical weights
    category_weights: Dict[int, float] = dataclasses.field(default_factory=lambda: {
        1: 2.50,  # Artist
        4: 2.00,  # Character
        3: 1.80,  # Copyright
        5: 1.30,  # Species
        0: 1.00,  # General
        7: 0.05,  # Meta
        2: 0.00,  # Contributor
        6: 0.00,  # Invalid
        8: 0.00,  # Lore
    })
    post2vec_batch: int = 200_000
    post2vec_index_sq8: bool = True

    # collab & taste personas
    fav_user_min_posts: int = 3
    fav_user_max_posts: int = 500
    cofav_min_weight: float = 0.5
    cofav_max_edges_per_post: int = 64
    taste_archetypes_k: int = 64
    taste_archetypes_top_posts: int = 500

    def __post_init__(self) -> None:
        if self.uploaders_csv is None:
            self.uploaders_csv = self.root / "uploaders_uploads.csv"
        if self.user_favorites_csv is None:
            self.user_favorites_csv = self.root / "user_favorites.csv"
        if self.post_cofav_parquet is None:
            self.post_cofav_parquet = self.features_dir / "post_cofav.parquet"
        if self.taste_archetypes_parquet is None:
            self.taste_archetypes_parquet = self.features_dir / "taste_archetypes.parquet"
        if self.taste_centroids_npy is None:
            self.taste_centroids_npy = self.features_dir / "taste_centroids.npy"

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2, default=str)