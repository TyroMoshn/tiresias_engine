from __future__ import annotations
"""
Pools stage:
- Parse pools.csv -> long parquet (pool_id, post_id), plus pools_meta.parquet
- Compute tag entropy per pool to filter noisy collections
- Build post-post edges aggregated across eligible pools with weight = 1/sqrt(|pool|)
- Build tag-tag co-occurrence from pools (optional, top-M tags per pool) with same weight
- Compute per-post count of pool inclusions and (if mmaps/post_ids.bin exists) write a memmap
"""

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from math import sqrt, log
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import polars as pl
from tqdm import tqdm

from .config import Config
from .utils import ensure_dir, newer_than, log


# ----------------------- helpers -----------------------

def _parse_post_ids_field(s: str) -> List[int]:
    # expects "{1,2,3}" or "{}"
    if not s or s == "{}":
        return []
    s = s.strip()
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1]
    if not s:
        return []
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(int(tok))
        except ValueError:
            # ignore garbage entries quietly
            continue
    # de-dup & keep deterministic order
    return sorted(set(out))


def _pool_weight(n: int) -> float:
    # weight per pair inside this pool
    return 1.0 / sqrt(max(1, n))


# ----------------------- steps -----------------------

def step_pools_parse(cfg: Config) -> None:
    """
    Read pools.csv, filter is_active=='t', size in [min,max],
    and expand to long format parquet partitioned by pool_shard.
    Also write pools_meta.parquet with per-pool metadata & size.
    """
    out_long = cfg.pools_parquet
    out_meta = cfg.pools_meta_parquet
    ensure_dir(out_long)

    # sentinel logic
    sentinel = out_long / "_SUCCESS"
    if sentinel.exists() and out_meta.exists() and not cfg.force and newer_than(
        sentinel, cfg.pools_csv
    ):
        log("[pools] already fresh - skip parse")
        return

    log("[pools] reading pools.csv...")
    # load with minimal typing, then clean
    df = pl.read_csv(cfg.pools_csv, infer_schema_length=0, ignore_errors=True)

    # normalize column names expected by spec
    cols = {c.lower(): c for c in df.columns}
    def pick(name: str) -> str: return cols.get(name, name)

    # keep only useful columns
    need = [
        pick("id"), pick("name"), pick("created_at"), pick("updated_at"),
        pick("creator_id"), pick("description"), pick("is_active"),
        pick("category"), pick("post_ids"),
    ]
    df = df.select([c for c in need if c in df.columns])

    # rename to canonical names
    rename_map = {
        pick("id"): "pool_id",
        pick("name"): "name",
        pick("created_at"): "created_at",
        pick("updated_at"): "updated_at",
        pick("creator_id"): "creator_id",
        pick("description"): "description",
        pick("is_active"): "is_active",
        pick("category"): "category",
        pick("post_ids"): "post_ids_raw",
    }
    df = df.rename(rename_map)

    # basic casting & filters
    df = (df
          .with_columns([
              pl.col("pool_id").cast(pl.Int64, strict=False),
              pl.col("creator_id").cast(pl.Int64, strict=False),
              pl.col("is_active").cast(pl.Utf8),
              pl.col("category").cast(pl.Utf8).str.to_lowercase(),
              pl.col("post_ids_raw").cast(pl.Utf8),
          ])
          .filter(pl.col("is_active") == "t")
    )

    # parse post_ids into list<int>
    df = df.with_columns(
        pl.col("post_ids_raw").map_elements(_parse_post_ids_field, return_dtype=pl.List(pl.Int64)).alias("post_ids")
    )

    # size & size filter
    df = df.with_columns(pl.col("post_ids").list.len().alias("size"))
    df = df.filter((pl.col("size") >= cfg.pool_min_size) & (pl.col("size") <= cfg.pool_max_size))

    # write meta (no huge arrays)
    meta = df.drop(["post_ids_raw"]).drop(["post_ids"])
    meta.write_parquet(out_meta, compression="zstd")

    # explode to long and shard by pool_id
    long = (df
            .select([pl.col("pool_id"), pl.col("post_ids")])
            .explode("post_ids")
            .rename({"post_ids": "post_id"})
            .with_columns([
                pl.col("pool_id").cast(pl.Int64),
                pl.col("post_id").cast(pl.Int64),
                (pl.col("pool_id") // 1000).alias("pool_shard"),
            ]))

    # (re)create directory and write partitioned
    if cfg.pools_parquet.exists():
        # remove old content to avoid stale partitions
        import shutil
        shutil.rmtree(cfg.pools_parquet, ignore_errors=True)
        ensure_dir(cfg.pools_parquet)

    long.write_parquet(
        cfg.pools_parquet.as_posix(),
        compression="zstd",
        use_pyarrow=True,
        pyarrow_options={
            "partition_cols": ["pool_shard"],
            "max_open_files": 4096,
            "max_partitions": 4096,
        },
    )
    sentinel.write_text("ok")
    log(f"[pools] parsed -> {out_meta.name} & long parquet")


def _load_tag_idf_if_available(cfg: Config) -> pl.DataFrame | None:
    # Try to load tag idf if stats already computed; else None.
    if cfg.tags_parquet.exists():
        try:
            df = pl.read_parquet(cfg.tags_parquet).select(["tag_id", "idf"])
            return df
        except Exception:
            return None
    return None


def step_pools_entropy(cfg: Config) -> None:
    """
    For each pool: compute tag-distribution entropy based on post_tags.
    If tags_parquet with idf exists, also compute idf-weighted entropy proxy.
    Save: pool_entropy.parquet with columns:
      pool_id, size, category, creator_id, H_tags, H_idf (nullable)
    """
    out = cfg.pools_entropy_parquet
    meta = cfg.pools_meta_parquet
    long_dir = cfg.pools_parquet
    pt_dir = cfg.post_tags_parquet

    if out.exists() and not cfg.force and newer_than(out, meta, long_dir / "_SUCCESS", pt_dir / "_SUCCESS"):
        log("[pools] entropy is fresh - skip")
        return

    log("[pools] computing tag entropy per pool...")
    # pools long: pool_id, post_id
    pools = pl.scan_parquet(f"{long_dir.as_posix()}/**/*.parquet").select(["pool_id", "post_id"])

    # post_tags: post_id, tag_id
    post_tags = pl.scan_parquet(f"{pt_dir.as_posix()}/**/*.parquet").select(["post_id", "tag_id"])

    # join and count tag frequency per pool
    pool_tags = pools.join(post_tags, on="post_id", how="inner") \
                     .group_by(["pool_id", "tag_id"]).agg(pl.len().alias("cnt"))

    # entropy H = -sum p*log(p), where p = cnt / sum(cnt) over tags in pool
    tag_sums = pool_tags.group_by("pool_id").agg(pl.col("cnt").sum().alias("total"))
    joined = pool_tags.join(tag_sums, on="pool_id", how="inner") \
                      .with_columns((pl.col("cnt") / pl.col("total")).alias("p")) \
                      .with_columns((-pl.col("p") * pl.col("p").log()).alias("term"))
    H = joined.group_by("pool_id").agg(pl.col("term").sum().alias("H_tags"))

    # optional: IDF-weighted "entropy" proxy
    idf_tbl = _load_tag_idf_if_available(cfg)
    if idf_tbl is not None:
        jt = pool_tags.join(idf_tbl.lazy(), on="tag_id", how="left").with_columns(
            pl.col("idf").fill_null(1.0)
        )
        jt = jt.with_columns((pl.col("cnt") * pl.col("idf")).alias("wcnt"))
        w_sums = jt.group_by("pool_id").agg(pl.col("wcnt").sum().alias("wtotal"))
        wj = jt.join(w_sums, on="pool_id", how="inner") \
               .with_columns((pl.col("wcnt") / pl.col("wtotal")).alias("wp")) \
               .with_columns((-pl.col("wp") * pl.col("wp").log()).alias("wterm"))
        H_idf = wj.group_by("pool_id").agg(pl.col("wterm").sum().alias("H_idf"))
        H = H.join(H_idf, on="pool_id", how="left")

    # attach meta (size, category, creator_id)
    meta_df = pl.read_parquet(meta).select(["pool_id", "size", "category", "creator_id"])
    out_df = meta_df.join(H.collect(), on="pool_id", how="left")

    out_df.write_parquet(out, compression="zstd")
    log("[pools] pool_entropy.parquet ready")


def _eligible_pools_mask(cfg: Config, meta_entropy: pl.DataFrame) -> pl.Series:
    """
    Decide which pools are eligible for generating edges/coocc:
    - keep category=='series' if pools_use_series
    - keep category=='collection' if pools_use_collections and entropy<=threshold
      where entropy is H_idf if available else H_tags normalized by log(V)
    """
    use_series = cfg.pools_use_series
    use_collections = cfg.pools_use_collections
    thr = float(cfg.pools_collection_entropy_max)

    # choose score
    has_idf = "H_idf" in meta_entropy.columns
    score = pl.when(pl.col("H_idf").is_not_null()).then(pl.col("H_idf")).otherwise(pl.col("H_tags")) if has_idf else pl.col("H_tags")

    mask_series = (pl.col("category") == "series") & pl.lit(use_series)
    mask_coll = (pl.col("category") == "collection") & pl.lit(use_collections) & (score <= thr)
    return mask_series | mask_coll


def step_pool_edges(cfg: Config) -> None:
    """
    Build aggregated post-post weighted edges from eligible pools.
    Output: pool_edges.parquet with columns [a_post_id, b_post_id, weight, pools]
    """
    out = cfg.pools_edges_parquet
    long_dir = cfg.pools_parquet
    meta_entropy = pl.read_parquet(cfg.pools_entropy_parquet)

    if out.exists() and not cfg.force and newer_than(out, long_dir / "_SUCCESS", cfg.pools_entropy_parquet):
        log("[pools] edges fresh - skip")
        return

    elig = meta_entropy.filter(_eligible_pools_mask(cfg, meta_entropy)).select(["pool_id", "size"])
    elig_ids = set(elig.get_column("pool_id").to_list())
    size_map = dict(elig.iter_rows())  # pool_id -> size

    log(f"[pools] edges: eligible pools = {len(elig_ids):,}")

    # iterate by shards to control memory
    shard_paths = sorted(Path(long_dir).glob("pool_shard=*/"))
    agg: Dict[Tuple[int, int], Tuple[float, int]] = defaultdict(lambda: (0.0, 0))

    def process_chunk(tbl: pl.DataFrame) -> None:
        nonlocal agg
        grp = tbl.group_by("pool_id").agg(pl.col("post_id")).iter_rows()
        for pool_id, posts in grp:
            if pool_id not in elig_ids:
                continue
            posts = sorted(set(int(p) for p in posts if p is not None))
            m = len(posts)
            if m < 2:
                continue
            w = _pool_weight(size_map.get(pool_id, m))
            for a, b in combinations(posts, 2):
                if a > b:
                    a, b = b, a
                weight, cnt = agg[(a, b)]
                agg[(a, b)] = (weight + w, cnt + 1)

    for shard in tqdm(shard_paths, desc="edges-shards"):
        lf = pl.scan_parquet(f"{shard.as_posix()}/**/*.parquet").select(["pool_id", "post_id"])
        df = lf.collect()
        if df.is_empty():
            continue
        process_chunk(df)

    # dump aggregated edges
    rows = [(a, b, w, c) for (a, b), (w, c) in agg.items()]
    if rows:
        pl.DataFrame(rows, schema={
            "a_post_id": pl.Int64, "b_post_id": pl.Int64,
            "weight": pl.Float32, "pools": pl.Int32
        },
        orient="row"
        ).write_parquet(out, compression="zstd")
    else:
        pl.DataFrame({"a_post_id": [], "b_post_id": [], "weight": [], "pools": []}).write_parquet(out, compression="zstd")

    log(f"[pools] edges written: {len(rows):,} pairs")


def step_post_in_pools_count(cfg: Config) -> None:
    """
    Count how many eligible pools each post participates in.
    If mmaps/post_ids.bin exists, write aligned int32 memmap: post_in_pools_count.bin
    """
    out_bin = cfg.mmaps_dir / "post_in_pools_count.bin"
    long_dir = cfg.pools_parquet

    meta_entropy = pl.read_parquet(cfg.pools_entropy_parquet)
    elig = meta_entropy.filter(_eligible_pools_mask(cfg, meta_entropy)).select(["pool_id"])
    elig_ids = set(elig.get_column("pool_id").to_list())

    # Count per post
    counts: Dict[int, int] = defaultdict(int)
    shard_paths = sorted(Path(long_dir).glob("pool_shard=*/"))
    for shard in tqdm(shard_paths, desc="post-counts"):
        tbl = pl.scan_parquet(f"{shard.as_posix()}/**/*.parquet").select(["pool_id", "post_id"]).collect()
        if tbl.is_empty(): continue
        for pool_id, post_id in tbl.iter_rows():
            if pool_id in elig_ids and post_id is not None:
                counts[int(post_id)] += 1

    # If mmaps/post_ids exists align, else just log summary
    post_ids_path = cfg.root / "mmaps" / "post_ids.bin"
    if post_ids_path.exists():
        post_ids = np.memmap(post_ids_path, mode="r", dtype=np.int64)
        arr = np.zeros_like(post_ids, dtype=np.int32)
        # post_ids are sorted (as per mmaps stage)
        # We'll use a dict lookup
        pos = {int(pid): idx for idx, pid in enumerate(post_ids)}
        for pid, c in counts.items():
            idx = pos.get(pid)
            if idx is not None:
                arr[idx] = int(c)
        mm = np.memmap(out_bin, mode="w+", dtype=np.int32, shape=arr.shape)
        mm[:] = arr
        mm.flush()
        log(f"[pools] post_in_pools_count.bin ready (aligned to mmaps/post_ids.bin)")
    else:
        log("[pools] mmaps/post_ids.bin not found - skipping memmap write (run mmaps stage first)")


def step_pool_tag_co(cfg: Config) -> None:
    """
    Build tag-tag co-occurrence from eligible pools.
    For each pool, take top-M tags (by frequency; if idf available, weight by idf*freq)
    and add weight 1/sqrt(|pool|) to each tag pair.
    Output: tag_co_from_pools.parquet [a_tag_id, b_tag_id, weight, pools]
    """
    out = cfg.tag_co_from_pools_parquet
    long_dir = cfg.pools_parquet
    pt_dir = cfg.post_tags_parquet

    if out.exists() and not cfg.force and newer_than(out, long_dir / "_SUCCESS", cfg.pools_entropy_parquet, pt_dir / "_SUCCESS"):
        log("[pools] tag co-occurrence fresh - skip")
        return

    meta_entropy = pl.read_parquet(cfg.pools_entropy_parquet)
    elig = meta_entropy.filter(_eligible_pools_mask(cfg, meta_entropy)).select(["pool_id", "size"])
    elig_ids = set(elig.get_column("pool_id").to_list())
    size_map = dict(elig.iter_rows())

    log(f"[pools] tag-co: eligible pools = {len(elig_ids):,}; top-M per pool = {cfg.pools_top_tags}")

    pools = pl.scan_parquet(f"{long_dir.as_posix()}/**/*.parquet").select(["pool_id", "post_id"])
    post_tags = pl.scan_parquet(f"{pt_dir.as_posix()}/**/*.parquet").select(["post_id", "tag_id"])

    # join to get tag occurrences per pool
    pt = pools.join(post_tags, on="post_id", how="inner").group_by(["pool_id", "tag_id"]).agg(pl.len().alias("cnt"))

    idf_tbl = _load_tag_idf_if_available(cfg)
    if idf_tbl is not None:
        pt = pt.join(idf_tbl.lazy(), on="tag_id", how="left").with_columns(pl.col("idf").fill_null(1.0))
        pt = pt.with_columns((pl.col("cnt") * pl.col("idf")).alias("score"))
    else:
        pt = pt.with_columns(pl.col("cnt").alias("score"))

    # For each pool: take top-M tag_ids by score, then build pairs
    df = pt.collect()
    df = df.filter(pl.col("pool_id").is_in(list(elig_ids)))
    grouped = df.sort(["pool_id", "score"], descending=[False, True]).group_by("pool_id").agg([
        pl.col("tag_id").head(cfg.pools_top_tags).alias("top_tags")
    ])

    acc: Dict[Tuple[int, int], Tuple[float, int]] = defaultdict(lambda: (0.0, 0))
    for pool_id, top_tags in tqdm(grouped.iter_rows(), total=grouped.height, desc="tag-co"):
        tags = [int(t) for t in top_tags if t is not None]
        if len(tags) < 2:
            continue
        w = _pool_weight(size_map.get(pool_id, len(tags)))
        for a, b in combinations(sorted(set(tags)), 2):
            if a > b:
                a, b = b, a
            weight, cnt = acc[(a, b)]
            acc[(a, b)] = (weight + w, cnt + 1)

    rows = [(a, b, w, c) for (a, b), (w, c) in acc.items()]
    if rows:
        pl.DataFrame(rows, schema={
            "a_tag_id": pl.Int32, "b_tag_id": pl.Int32,
            "weight": pl.Float32, "pools": pl.Int32
        },
        orient="row"
        ).write_parquet(out, compression="zstd")
    else:
        pl.DataFrame({"a_tag_id": [], "b_tag_id": [], "weight": [], "pools": []}).write_parquet(out, compression="zstd")

    log(f"[pools] tag_co_from_pools.parquet written: {len(rows):,} pairs")
