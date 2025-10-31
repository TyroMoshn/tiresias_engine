# build_index/main.py
from __future__ import annotations
import argparse, os
from pathlib import Path
from .config import Config
from .utils import log
from .io_stage import step_parquet
from .tags_stage import step_tags_and_post_tags, step_implications
from .index_stage import step_build_bitmaps, step_build_mmaps
from .stats_stage import step_tag_stats, step_pmi, step_topk
from .tag2vec_stage import step_tag2vec

from .pools_stage import (
    step_pools_parse, step_pools_entropy, step_pool_edges,
    step_post_in_pools_count, step_pool_tag_co
)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Конвертация и препроцессинг БД")
    p.add_argument('--root', type=Path, required=True, help='Корень data/')
    p.add_argument('--csv', type=Path, default=None, help='Путь к db.csv (если не data/db.csv)')
    p.add_argument('--tags-csv', type=Path, default=None, help='Путь к tags.csv (если не data/tags.csv)')
    p.add_argument('--tag-aliases', type=Path, default=None, help='Путь к tag_aliases.csv (если не data/tag_aliases.csv)')
    p.add_argument('--tag-implications', type=Path, default=None, help='Путь к tag_implications.csv (если не data/tag_implications.csv)')
    p.add_argument('--workers', type=int, default=os.cpu_count() or 4)
    p.add_argument('--force', action='store_true', help='Пересчитывать даже если свежее')
    p.add_argument('--do', nargs='+', default=[
        'parquet', 'tags', 'implications', 'post_tags', 'bitmaps', 'mmaps', 'stats', 'pmi', 'topk'
    ], help='Какие шаги выполнить')
    # tuning
    p.add_argument('--topk-k', type=int, default=5000)
    p.add_argument('--pmi-support', type=int, default=50)
    p.add_argument('--pmi-top-m-per-post', type=int, default=16)
    p.add_argument('--idf-alpha', type=float, default=1000.0)
    p.add_argument('--anc-cache-depth', type=int, default=3)
    p.add_argument('--anc-cache-top-percent', type=float, default=0.05)
    p.add_argument('--roar-shard-size', type=int, default=0)
    p.add_argument("--topk-mode", choices=["static","sqrt_df"], default="static")
    p.add_argument("--topk-k-min", type=int, default=100)
    p.add_argument("--topk-k-max", type=int, default=5000)
    p.add_argument("--topk-beta", type=float, default=10.0)
    p.add_argument("--idf-source", choices=["tags","local","auto"], default="auto")
    p.add_argument("--idf-auto-threshold", type=float, default=0.90)
    p.add_argument("--reliable-only", dest="reliable_only", action="store_true", default=True)
    p.add_argument("--no-reliable-only", dest="reliable_only", action="store_false")
    # pools
    p.add_argument('--pools-csv', type=Path, default=None, help='Path to pools.csv (default: root/pools.csv)')
    p.add_argument('--pool-min-size', type=int, default=3)
    p.add_argument('--pool-max-size', type=int, default=200)
    p.add_argument('--pools-use-series', dest='pools_use_series', action='store_true', default=True)
    p.add_argument('--no-pools-use-series', dest='pools_use_series', action='store_false')
    p.add_argument('--pools-use-collections', dest='pools_use_collections', action='store_true', default=True)
    p.add_argument('--no-pools-use-collections', dest='pools_use_collections', action='store_false')
    p.add_argument('--pools-collection-entropy-max', type=float, default=6.0)
    p.add_argument('--pools-top-tags', type=int, default=24)
    # tag2vec
    p.add_argument("--tag2vec-dim", type=int, default=128)
    p.add_argument("--tag2vec-min-df", type=int, default=100)
    p.add_argument("--tag2vec-max-tags", type=int, default=200_000)
    p.add_argument("--tag2vec-source", choices=["merge","pmi","pools"], default="merge")
    p.add_argument("--tag2vec-pool-alpha", type=float, default=0.5)
    p.add_argument("--tag2vec-shift", type=float, default=0.0)
    p.add_argument("--tag2vec-knn-k", type=int, default=32)
    return p.parse_args()

def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    csv = args.csv or (root / 'db.csv')
    tags_csv = args.tags_csv or (root / 'tags.csv')
    tag_aliases_csv = args.tag_aliases or (root / 'tag_aliases.csv')
    tag_implications_csv = args.tag_implications or (root / 'tag_implications.csv')

    cfg = Config(
        root=root,
        csv=csv,
        posts_parquet=root / 'posts_parquet',
        post_tags_parquet=root / 'post_tags_parquet',
        bitmaps_dir=root / 'bitmaps',
        topk_dir=root / 'topk',
        features_dir=root / 'features',
        tags_parquet=root / 'tags.parquet',
        tags_csv=tags_csv,
        tag_aliases_csv=tag_aliases_csv,
        tag_implications_csv=tag_implications_csv,
        # pools start
        pools_csv = args.pools_csv or (root / 'pools.csv'),
        pools_parquet = root / 'pools_parquet',
        pools_meta_parquet = root / 'pools_meta.parquet',
        pools_entropy_parquet = root / 'pool_entropy.parquet',
        pools_edges_parquet = root / 'pool_edges.parquet',
        tag_co_from_pools_parquet = root / 'tag_co_from_pools.parquet',
        mmaps_dir = root / 'mmaps',
        pool_min_size = args.pool_min_size,
        pool_max_size = args.pool_max_size,
        pools_use_series = args.pools_use_series,
        pools_use_collections = args.pools_use_collections,
        pools_collection_entropy_max = args.pools_collection_entropy_max,
        pools_top_tags = args.pools_top_tags,
        # pools end
        workers=args.workers,
        force=args.force,
        topk_k=args.topk_k,
        pmi_support=args.pmi_support,
        pmi_top_m_per_post=args.pmi_top_m_per_post,
        idf_alpha=args.idf_alpha,
        anc_cache_depth=args.anc_cache_depth,
        anc_cache_top_percent=args.anc_cache_top_percent,
        roar_shard_size=args.roar_shard_size,
        topk_mode=args.topk_mode,
        topk_k_min=args.topk_k_min,
        topk_k_max=args.topk_k_max,
        topk_beta=args.topk_beta,
        idf_source=args.idf_source,
        idf_auto_switch_threshold=args.idf_auto_threshold,
        reliable_only=args.reliable_only,
    )

    log("CONFIG:\n" + cfg.to_json())
    steps = set(args.do)

    if 'parquet' in steps:
        step_parquet(cfg)

    if 'tags' in steps or 'post_tags' in steps:
        step_tags_and_post_tags(cfg)

    if 'implications' in steps:
        step_implications(cfg)

    if 'bitmaps' in steps:
        step_build_bitmaps(cfg)

    if 'mmaps' in steps:
        step_build_mmaps(cfg)

    if 'stats' in steps:
        step_tag_stats(cfg)

    if 'pmi' in steps:
        step_pmi(cfg)

    if 'topk' in steps:
        step_topk(cfg)
    # pools
    if 'pools' in steps or 'pools_parse' in steps:
        step_pools_parse(cfg)

    if 'pools' in steps or 'pools_entropy' in steps:
        step_pools_entropy(cfg)
        
    if 'pools' in steps or 'pools_edges' in steps:
        step_pool_edges(cfg)

    if 'pools' in steps or 'pools_post_counts' in steps:
        step_post_in_pools_count(cfg)

    if 'pools' in steps or 'pools_tag_co' in steps:
        step_pool_tag_co(cfg)
    
    if 'tag2vec' in steps:
        step_tag2vec(cfg)

    log("Готово.")

if __name__ == '__main__':
    main()