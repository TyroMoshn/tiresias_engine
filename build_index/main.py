# build_index/main.py
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Set, List

from .config import Config, get_default_data_root
from .utils import log
from .io_stage import step_parquet
from .tags_stage import step_tags_and_post_tags, step_implications
from .index_stage import step_build_bitmaps, step_build_mmaps
from .stats_stage import step_tag_stats, step_pmi, step_topk
from .tag2vec_stage import step_tag2vec
from .post2vec_stage import step_post2vec
from .uploaders_extract import step_uploaders_extract
from .pools_stage import (
    step_pools_parse,
    step_pools_entropy,
    step_pool_edges,
    step_post_in_pools_count,
    step_pool_tag_co,
)
from .collab_stage import (
    step_collab_favs,
    step_taste_archetypes,
)

# Canonical DAG order of pipeline execution
PIPELINE_ORDER = [
    "parquet",
    "tags",
    "post_tags",
    "implications",
    "mmaps",
    "bitmaps",
    "stats",
    "topk",
    "pools_parse",
    "pools_entropy",
    "pools_edges",
    "pools_post_counts",
    "pools_tag_co",
    "pmi",
    "uploaders",
    "tag2vec",
    "post2vec",
    "collab_favs",
    "taste_archetypes",
]


def validate_step_dependencies(step: str, cfg: Config) -> None:
    """Validate that prerequisite artifacts exist before executing a step."""
    if step == "bitmaps":
        if not (cfg.mmaps_dir / "post_ids.bin").exists():
            raise FileNotFoundError(
                "Cannot run 'bitmaps': required artifact 'mmaps/post_ids.bin' not found. "
                "Execute the 'mmaps' step first."
            )
        if not (cfg.post_tags_parquet / "_SUCCESS").exists():
            raise FileNotFoundError(
                "Cannot run 'bitmaps': required artifact 'post_tags_parquet/_SUCCESS' not found. "
                "Execute the 'tags' or 'post_tags' step first."
            )

    elif step == "stats":
        if not (cfg.posts_parquet / "_SUCCESS").exists():
            raise FileNotFoundError("Cannot run 'stats': 'posts_parquet' not found. Run 'parquet' first.")
        if not (cfg.root / "tags_dict.parquet").exists():
            raise FileNotFoundError("Cannot run 'stats': 'tags_dict.parquet' not found. Run 'tags' first.")
        if not (cfg.post_tags_parquet / "_SUCCESS").exists():
            raise FileNotFoundError("Cannot run 'stats': 'post_tags_parquet' not found. Run 'tags' first.")

    elif step == "topk":
        if not (cfg.posts_parquet / "_SUCCESS").exists():
            raise FileNotFoundError("Cannot run 'topk': 'posts_parquet' not found. Run 'parquet' first.")
        if not (cfg.post_tags_parquet / "_SUCCESS").exists():
            raise FileNotFoundError("Cannot run 'topk': 'post_tags_parquet' not found. Run 'tags' first.")

    elif step == "pools_entropy":
        if not (cfg.pools_parquet / "_SUCCESS").exists():
            raise FileNotFoundError("Cannot run 'pools_entropy': 'pools_parquet' not found. Run 'pools_parse' first.")

    elif step == "pools_post_counts":
        if not (cfg.mmaps_dir / "post_ids.bin").exists():
            raise FileNotFoundError(
                "Cannot run 'pools_post_counts': 'mmaps/post_ids.bin' not found. Run 'mmaps' first."
            )
        if not (cfg.pools_parquet / "_SUCCESS").exists():
            raise FileNotFoundError(
                "Cannot run 'pools_post_counts': 'pools_parquet' not found. Run 'pools_parse' first."
            )

    elif step == "pools_tag_co":
        if not (cfg.pools_parquet / "_SUCCESS").exists():
            raise FileNotFoundError("Cannot run 'pools_tag_co': 'pools_parquet' not found. Run 'pools_parse' first.")

    elif step == "pmi":
        if not (cfg.post_tags_parquet / "_SUCCESS").exists():
            raise FileNotFoundError("Cannot run 'pmi': 'post_tags_parquet' not found. Run 'tags' first.")

    elif step == "tag2vec":
        has_pmi = (cfg.root / "tag_pmi.parquet").exists()
        has_pools = cfg.tag_co_from_pools_parquet.exists()
        if not (has_pmi or has_pools):
            raise FileNotFoundError("Cannot run 'tag2vec': neither 'tag_pmi.parquet' nor pool co-occurrence exists.")

    elif step == "post2vec":
        if not (cfg.features_dir / "tag2vec.parquet").exists():
            raise FileNotFoundError("Cannot run 'post2vec': 'features/tag2vec.parquet' not found. Run 'tag2vec' first.")
        if not (cfg.post_tags_parquet / "_SUCCESS").exists():
            raise FileNotFoundError("Cannot run 'post2vec': 'post_tags_parquet' not found. Run 'tags' first.")

    elif step == "collab_favs":
        if not cfg.user_favorites_csv.exists():
            raise FileNotFoundError(
                f"Cannot run 'collab_favs': 'user_favorites.csv' not found at {cfg.user_favorites_csv}."
            )

    elif step == "taste_archetypes":
        if not cfg.user_favorites_csv.exists():
            raise FileNotFoundError(
                f"Cannot run 'taste_archetypes': 'user_favorites.csv' not found at {cfg.user_favorites_csv}."
            )
        if not (cfg.features_dir / "post2vec.parquet").exists():
            raise FileNotFoundError(
                "Cannot run 'taste_archetypes': 'features/post2vec.parquet' not found. Run 'post2vec' first."
            )


def parse_args() -> argparse.Namespace:
    default_root = get_default_data_root()
    p = argparse.ArgumentParser(description="TIRESIAS_ENGINE - Offline Indexing & Preprocessing Pipeline")
    p.add_argument("--root", type=Path, default=default_root, help="Root data directory (default: TIRESIAS_DATA_ROOT or ./data)")
    p.add_argument("--csv", type=Path, default=None, help="Path to posts.csv (default: <root>/posts.csv)")
    p.add_argument("--tags-csv", type=Path, default=None, help="Path to tags.csv (default: <root>/tags.csv)")
    p.add_argument("--tag-aliases", type=Path, default=None, help="Path to tag_aliases.csv (default: <root>/tag_aliases.csv)")
    p.add_argument("--tag-implications", type=Path, default=None, help="Path to tag_implications.csv (default: <root>/tag_implications.csv)")
    p.add_argument("--uploaders-csv", type=Path, default=None, help="Path to uploaders_uploads.csv (default: <root>/uploaders_uploads.csv)")
    p.add_argument("--user-favorites-csv", type=Path, default=None, help="Path to user_favorites.csv (default: <root>/user_favorites.csv)")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 4, help="Worker threads")
    p.add_argument("--force", action="store_true", help="Recalculate even if artifacts are fresh")
    p.add_argument(
        "--do",
        nargs="+",
        default=[
            "parquet", "tags", "implications", "mmaps", "bitmaps", "stats", "topk", "pmi"
        ],
        help="Pipeline steps to execute. Use 'all' to run complete pipeline in DAG order.",
    )
    # tuning
    p.add_argument("--topk-k", type=int, default=5000)
    p.add_argument("--pmi-support", type=int, default=50)
    p.add_argument("--pmi-top-m-per-post", type=int, default=16)
    p.add_argument("--idf-alpha", type=float, default=1000.0)
    p.add_argument("--anc-cache-depth", type=int, default=3)
    p.add_argument("--anc-cache-top-percent", type=float, default=0.05)
    p.add_argument("--roar-shard-size", type=int, default=0)
    p.add_argument("--topk-mode", choices=["static", "sqrt_df"], default="static")
    p.add_argument("--topk-k-min", type=int, default=100)
    p.add_argument("--topk-k-max", type=int, default=5000)
    p.add_argument("--topk-beta", type=float, default=10.0)
    p.add_argument("--idf-source", choices=["tags", "local", "auto"], default="auto")
    p.add_argument("--idf-auto-threshold", type=float, default=0.90)
    p.add_argument("--reliable-only", dest="reliable_only", action="store_true", default=True)
    p.add_argument("--no-reliable-only", dest="reliable_only", action="store_false")
    # pools
    p.add_argument("--pools-csv", type=Path, default=None, help="Path to pools.csv (default: <root>/pools.csv)")
    p.add_argument("--pool-min-size", type=int, default=3)
    p.add_argument("--pool-max-size", type=int, default=200)
    p.add_argument("--pools-use-series", dest="pools_use_series", action="store_true", default=True)
    p.add_argument("--no-pools-use-series", dest="pools_use_series", action="store_false")
    p.add_argument("--pools-use-collections", dest="pools_use_collections", action="store_true", default=True)
    p.add_argument("--no-pools-use-collections", dest="pools_use_collections", action="store_false")
    p.add_argument("--pools-collection-entropy-max", type=float, default=6.0)
    p.add_argument("--pools-top-tags", type=int, default=24)
    # tag2vec
    p.add_argument("--tag2vec-dim", type=int, default=128)
    p.add_argument("--tag2vec-min-df", type=int, default=100)
    p.add_argument("--tag2vec-max-tags", type=int, default=200_000)
    p.add_argument("--tag2vec-source", choices=["merge", "pmi", "pools"], default="merge")
    p.add_argument("--tag2vec-pool-alpha", type=float, default=0.5)
    p.add_argument("--tag2vec-shift", type=float, default=0.0)
    p.add_argument("--tag2vec-knn-k", type=int, default=32)
    # post2vec v2 & collab
    p.add_argument("--post2vec-batch", type=int, default=200_000)
    p.add_argument("--post2vec-sq8", dest="post2vec_sq8", action="store_true", default=True)
    p.add_argument("--taste-archetypes-k", type=int, default=64)
    p.add_argument("--taste-archetypes-top-posts", type=int, default=500)
    p.add_argument("--fav-user-min-posts", type=int, default=3)
    p.add_argument("--fav-user-max-posts", type=int, default=500)
    p.add_argument("--cofav-min-weight", type=float, default=0.5)
    p.add_argument("--cofav-max-edges-per-post", type=int, default=64)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    csv = args.csv or (root / "posts.csv")
    tags_csv = args.tags_csv or (root / "tags.csv")
    tag_aliases_csv = args.tag_aliases or (root / "tag_aliases.csv")
    tag_implications_csv = args.tag_implications or (root / "tag_implications.csv")
    uploaders_csv = args.uploaders_csv or (root / "uploaders_uploads.csv")
    user_favorites_csv = args.user_favorites_csv or (root / "user_favorites.csv")

    cfg = Config(
        root=root,
        csv=csv,
        posts_parquet=root / "posts_parquet",
        post_tags_parquet=root / "post_tags_parquet",
        bitmaps_dir=root / "bitmaps",
        topk_dir=root / "topk",
        features_dir=root / "features",
        tags_parquet=root / "tags.parquet",
        tags_csv=tags_csv,
        tag_aliases_csv=tag_aliases_csv,
        tag_implications_csv=tag_implications_csv,
        # pools
        pools_csv=args.pools_csv or (root / "pools.csv"),
        pools_parquet=root / "pools_parquet",
        pools_meta_parquet=root / "pools_meta.parquet",
        pools_entropy_parquet=root / "pool_entropy.parquet",
        pools_edges_parquet=root / "pool_edges.parquet",
        uploaders_csv=uploaders_csv,
        tag_co_from_pools_parquet=root / "tag_co_from_pools.parquet",
        mmaps_dir=root / "mmaps",
        # collab
        user_favorites_csv=user_favorites_csv,
        post_cofav_parquet=root / "features" / "post_cofav.parquet",
        taste_archetypes_parquet=root / "features" / "taste_archetypes.parquet",
        taste_centroids_npy=root / "features" / "taste_centroids.npy",
        pool_min_size=args.pool_min_size,
        pool_max_size=args.pool_max_size,
        pools_use_series=args.pools_use_series,
        pools_use_collections=args.pools_use_collections,
        pools_collection_entropy_max=args.pools_collection_entropy_max,
        pools_top_tags=args.pools_top_tags,
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
        post2vec_batch=args.post2vec_batch,
        post2vec_index_sq8=args.post2vec_sq8,
        taste_archetypes_k=args.taste_archetypes_k,
        taste_archetypes_top_posts=args.taste_archetypes_top_posts,
        fav_user_min_posts=args.fav_user_min_posts,
        fav_user_max_posts=args.fav_user_max_posts,
        cofav_min_weight=args.cofav_min_weight,
        cofav_max_edges_per_post=args.cofav_max_edges_per_post,
    )

    log("CONFIG:\n" + cfg.to_json())

    raw_steps = set(args.do)
    if "all" in raw_steps:
        steps_to_run = set(PIPELINE_ORDER)
    else:
        steps_to_run = raw_steps

    # Map aliases
    if "pools" in steps_to_run:
        steps_to_run.update(["pools_parse", "pools_entropy", "pools_edges", "pools_post_counts", "pools_tag_co"])
    if "tags" in steps_to_run:
        steps_to_run.add("post_tags")
    if "collab" in steps_to_run:
        steps_to_run.update(["collab_favs", "taste_archetypes"])

    # Execute in strict topological DAG order
    for step in PIPELINE_ORDER:
        if step not in steps_to_run:
            continue

        validate_step_dependencies(step, cfg)

        if step == "parquet":
            step_parquet(cfg)
        elif step in ("tags", "post_tags"):
            # Handled jointly by step_tags_and_post_tags
            step_tags_and_post_tags(cfg)
            steps_to_run.discard("post_tags")
            steps_to_run.discard("tags")
        elif step == "implications":
            step_implications(cfg)
        elif step == "mmaps":
            step_build_mmaps(cfg)
        elif step == "bitmaps":
            step_build_bitmaps(cfg)
        elif step == "stats":
            step_tag_stats(cfg)
        elif step == "topk":
            step_topk(cfg)
        elif step == "pools_parse":
            step_pools_parse(cfg)
        elif step == "pools_entropy":
            step_pools_entropy(cfg)
        elif step == "pools_edges":
            step_pool_edges(cfg)
        elif step == "pools_post_counts":
            step_post_in_pools_count(cfg)
        elif step == "pools_tag_co":
            step_pool_tag_co(cfg)
        elif step == "pmi":
            step_pmi(cfg)
        elif step == "uploaders":
            step_uploaders_extract(cfg)
        elif step == "tag2vec":
            step_tag2vec(cfg)
        elif step == "post2vec":
            step_post2vec(cfg)
        elif step == "collab_favs":
            step_collab_favs(cfg)
        elif step == "taste_archetypes":
            step_taste_archetypes(cfg)

    log("Pipeline execution finished successfully.")


if __name__ == "__main__":
    main()
