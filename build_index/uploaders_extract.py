#!/usr/bin/env python3
"""
Convert uploader information from posts into uploaders_uploads.csv.

Output schema:
    user_id (int), upload_count (int), uploads (string "{id,id,...}")
Sorted by upload_count ascending.

Provides:
- step_uploaders_extract(cfg: Config) for pipeline integration
- CLI interface for standalone invocation.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

# Local imports
try:
    from .config import Config, get_default_data_root
    from .utils import log, ensure_dir, newer_than
except ImportError:
    # Direct script execution fallback
    HERE = Path(__file__).resolve().parent
    PROJECT_ROOT = HERE.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from build_index.config import Config, get_default_data_root
    from build_index.utils import log, ensure_dir, newer_than


def extract_with_duckdb(src_path: Path, is_parquet: bool, out_path: Path, workers: int = 4) -> None:
    """Fast vectorized extraction using DuckDB SQL engine."""
    import duckdb

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={workers}")

    if is_parquet:
        source_expr = f"parquet_scan('{src_path.as_posix()}/**/*.parquet')"
    else:
        source_expr = f"read_csv_auto('{src_path.as_posix()}', ignore_errors=true)"

    query = f"""
    COPY (
        WITH aggregated AS (
            SELECT
                CAST(uploader_id AS BIGINT) AS user_id,
                COUNT(CAST(id AS BIGINT)) AS upload_count,
                '{{' || string_agg(CAST(id AS VARCHAR), ',' ORDER BY CAST(id AS BIGINT)) || '}}' AS uploads
            FROM {source_expr}
            WHERE uploader_id IS NOT NULL AND id IS NOT NULL
            GROUP BY uploader_id
        )
        SELECT user_id, upload_count, uploads
        FROM aggregated
        ORDER BY upload_count ASC
    ) TO '{out_path.as_posix()}' (HEADER, DELIMITER ',')
    """
    con.execute(query)


def extract_with_polars(src_path: Path, is_parquet: bool, out_path: Path) -> None:
    """Fallback extraction using Polars streaming."""
    import polars as pl

    if is_parquet:
        lf = pl.scan_parquet(f"{src_path.as_posix()}/**/*.parquet")
    else:
        lf = pl.scan_csv(src_path, ignore_errors=True)

    agg = (
        lf.select([
            pl.col("uploader_id").cast(pl.Int64, strict=False),
            pl.col("id").cast(pl.Int64, strict=False),
        ])
        .drop_nulls()
        .group_by("uploader_id")
        .agg([
            pl.len().alias("upload_count"),
            pl.concat_str([
                pl.lit("{"),
                pl.col("id").sort().cast(pl.Utf8).str.concat(","),
                pl.lit("}"),
            ]).alias("uploads")
        ])
        .rename({"uploader_id": "user_id"})
        .sort("upload_count", descending=False)
        .collect(engine="streaming")
    )

    agg.write_csv(out_path)


def step_uploaders_extract(cfg: Config) -> None:
    """Pipeline step: extract uploaders_uploads.csv."""
    out_path = cfg.uploaders_csv
    ensure_dir(out_path.parent)

    source_is_parquet = False
    source_path = cfg.posts_parquet

    if (cfg.posts_parquet / "_SUCCESS").exists():
        source_is_parquet = True
        dep = cfg.posts_parquet / "_SUCCESS"
    elif cfg.csv.exists():
        source_path = cfg.csv
        dep = cfg.csv
    else:
        raise FileNotFoundError(
            f"Cannot extract uploaders: neither {cfg.posts_parquet} nor {cfg.csv} exists."
        )

    if out_path.exists() and not cfg.force and newer_than(out_path, dep):
        log("[uploaders] already fresh - skip")
        return

    log(f"[uploaders] extracting from {'posts_parquet' if source_is_parquet else 'posts.csv'} -> {out_path}...")

    # Attempt DuckDB first for maximum throughput, falling back to Polars
    try:
        extract_with_duckdb(source_path, source_is_parquet, out_path, workers=cfg.workers)
        log("[uploaders] extraction completed via DuckDB.")
    except Exception as duck_err:
        log(f"[uploaders] DuckDB extraction skipped ({duck_err}), trying Polars streaming...")
        extract_with_polars(source_path, source_is_parquet, out_path)
        log("[uploaders] extraction completed via Polars.")


def main() -> None:
    default_root = get_default_data_root()
    p = argparse.ArgumentParser(description="Extract uploaders_uploads.csv from posts")
    p.add_argument("--root", type=Path, default=default_root, help="Root data directory")
    p.add_argument("--csv", type=Path, default=None, help="Path to posts.csv (default: <root>/posts.csv)")
    p.add_argument("--out", type=Path, default=None, help="Output path (default: <root>/uploaders_uploads.csv)")
    p.add_argument("--workers", type=int, default=4, help="Worker threads")
    p.add_argument("--force", action="store_true", help="Force recalculation")
    args = p.parse_args()

    root = args.root.resolve()
    csv_path = args.csv or (root / "posts.csv")
    out_path = args.out or (root / "uploaders_uploads.csv")

    cfg = Config(
        root=root,
        csv=csv_path,
        posts_parquet=root / "posts_parquet",
        post_tags_parquet=root / "post_tags_parquet",
        bitmaps_dir=root / "bitmaps",
        topk_dir=root / "topk",
        features_dir=root / "features",
        tags_parquet=root / "tags.parquet",
        tags_csv=root / "tags.csv",
        tag_aliases_csv=root / "tag_aliases.csv",
        tag_implications_csv=root / "tag_implications.csv",
        pools_csv=root / "pools.csv",
        pools_parquet=root / "pools_parquet",
        pools_meta_parquet=root / "pools_meta.parquet",
        pools_entropy_parquet=root / "pool_entropy.parquet",
        pools_edges_parquet=root / "pool_edges.parquet",
        uploaders_csv=out_path,
        tag_co_from_pools_parquet=root / "tag_co_from_pools.parquet",
        mmaps_dir=root / "mmaps",
        workers=args.workers,
        force=args.force,
    )

    step_uploaders_extract(cfg)


if __name__ == "__main__":
    main()
