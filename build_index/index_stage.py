# build_index/index_stage.py
from __future__ import annotations
import math, struct
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
import polars as pl
from pyroaring import BitMap
from tqdm import tqdm

from .config import Config
from .utils import ensure_dir, newer_than, log

def _write_roar(path: Path, ids: np.ndarray) -> None:
    bm = BitMap(ids.tolist())
    with open(path, 'wb') as f:
        f.write(bm.serialize())


def _write_roar_shard(out_pack: Path, out_index: Path, shard_pairs: List[Tuple[int, np.ndarray]]) -> None:
    # shard_pairs: [(tag_id, ids), ...]
    offset = 0
    index_rows: List[Tuple[int,int,int]] = []  # (tag_id, offset, length)
    buf = bytearray()
    for tag_id, ids in shard_pairs:
        bm = BitMap(ids.tolist())
        blob = bm.serialize()
        buf += blob
        index_rows.append((int(tag_id), offset, len(blob)))
        offset += len(blob)
    with open(out_pack, 'wb') as f:
        f.write(buf)
    with open(out_index, 'wb') as f:
        for tag_id, off, ln in index_rows:
            f.write(struct.pack('<iqi', int(tag_id), off, ln))

def step_build_bitmaps(cfg: Config) -> None:
    ensure_dir(cfg.bitmaps_dir)
    sentinel = cfg.bitmaps_dir / "_SUCCESS"
    if sentinel.exists() and not cfg.force and newer_than(sentinel, cfg.post_tags_parquet / "_SUCCESS"):
        log("[bitmaps] уже актуально — пропуск")
        return

    log("[bitmaps] строим roaring-индексы…")
    shard_dirs = sorted(p for p in cfg.post_tags_parquet.glob("tag_shard=*") if p.is_dir())
    legacy_tag_dirs = sorted(p for p in cfg.post_tags_parquet.glob("tag_id=*") if p.is_dir())

    if shard_dirs:
        # Современный режим: tag_shard=*
        def process_shard(shard_dir: Path) -> int:
            tbl = pl.scan_parquet(f"{shard_dir.as_posix()}/**/*.parquet").select(["post_id","tag_id"]).collect()
            if tbl.is_empty():
                return 0

            grouped = tbl.group_by("tag_id").agg(pl.col("post_id")).sort("tag_id")
            shard_id = int(shard_dir.name.split('=')[1])

            pairs = []
            for tag_id, post_ids in grouped.iter_rows():
                ids = np.array(post_ids, dtype=np.int32)
                ids.sort()
                pairs.append((int(tag_id), ids))

            out_pack  = cfg.bitmaps_dir / f"shard_{shard_id:04d}.roarpack"
            out_index = cfg.bitmaps_dir / f"index_{shard_id:04d}.bin"
            _write_roar_shard(out_pack, out_index, pairs)
            return grouped.height

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=cfg.workers) as ex:
            list(tqdm(ex.map(process_shard, shard_dirs), total=len(shard_dirs)))

    else:
        # Legacy-режим: старые tag_id=*
        tag_parts = legacy_tag_dirs

        def process_part(part_dir: Path) -> tuple[int, np.ndarray]:
            tag_id = int(part_dir.name.split('=')[1])
            tbl = pl.scan_parquet(f"{part_dir.as_posix()}/*.parquet").collect()
            ids = tbl.get_column("post_id").to_numpy().astype(np.int32)
            ids.sort()
            return (tag_id, ids)

        acc: list[tuple[int, np.ndarray]] = []
        for part_dir in tqdm(tag_parts, desc="collect"):
            tag_id, ids = process_part(part_dir)
            if ids is not None:
                acc.append((tag_id, ids))

        acc.sort(key=lambda t: t[0])
        shard = cfg.roar_shard_size if cfg.roar_shard_size > 0 else 10000  # подстраховка
        for i in tqdm(range(0, len(acc), shard), desc="shard"):
            chunk = acc[i:i+shard]
            if not chunk:
                continue
            shard_id = i // shard
            out_pack  = cfg.bitmaps_dir / f"shard_{shard_id:04d}.roarpack"
            out_index = cfg.bitmaps_dir / f"index_{shard_id:04d}.bin"
            pairs = [(int(tag_id), ids) for tag_id, ids in chunk]
            _write_roar_shard(out_pack, out_index, pairs)

    sentinel.write_text("ok")
    log("[bitmaps] готово")

def step_build_mmaps(cfg: Config) -> None:
    meta_dir = cfg.root / "mmaps"
    ensure_dir(meta_dir)
    sentinel = meta_dir / "_SUCCESS"
    if sentinel.exists() and not cfg.force and newer_than(sentinel, cfg.posts_parquet / "_SUCCESS"):
        log("[mmaps] уже актуально — пропуск")
        return

    log("[mmaps] собираем компактные столбцы…")
    scan = pl.scan_parquet(f"{cfg.posts_parquet.as_posix()}/**/*.parquet", hive_partitioning=True)
    try:
        df = scan.select([
            pl.col("id").cast(pl.Int64),
            pl.col("rating"),
            (pl.col("created_at").cast(pl.Datetime).dt.date().cast(pl.Int32)).alias("epoch_day"),
            pl.col("image_width").cast(pl.Int32).alias("w"),
            pl.col("image_height").cast(pl.Int32).alias("h"),
            pl.col("score").cast(pl.Int32),
            pl.col("fav_count").cast(pl.Int32),
            pl.col("is_deleted").cast(pl.Boolean),
            pl.col("is_pending").cast(pl.Boolean),
        ]).collect(engine="streaming")
    except Exception:
        df = scan.select([
            pl.col("id").cast(pl.Int64),
            (pl.col("created_at").cast(pl.Datetime).dt.date().cast(pl.Int32)).alias("epoch_day"),
            pl.col("image_width").cast(pl.Int32).alias("w"),
            pl.col("image_height").cast(pl.Int32).alias("h"),
            pl.col("score").cast(pl.Int32),
            pl.col("fav_count").cast(pl.Int32),
            pl.col("is_deleted").cast(pl.Boolean),
            pl.col("is_pending").cast(pl.Boolean),
        ]).collect(engine="streaming")
        df = df.with_columns(pl.lit(None).alias("rating"))

    df = df.sort("id")
    post_ids = df.get_column("id").to_numpy()
    n = post_ids.size

    try:
        for r in ("s", "q", "e"):
            mask_ids = post_ids[df.get_column("rating").to_numpy() == r].astype(np.int32)
            _write_roar(meta_dir / f"rating_{r}.roar", mask_ids)
    except Exception:
        for r in ("s", "q", "e"):
            rp = f"{cfg.posts_parquet.as_posix()}/rating={r}/**/*.parquet"
            if list(Path(cfg.posts_parquet).glob(f"rating={r}")):
                ids_r = pl.scan_parquet(rp, hive_partitioning=True).select(pl.col("id").cast(pl.Int64)).collect()
                mask_ids = np.array(ids_r.get_column("id").to_list(), dtype=np.int32)
                mask_ids.sort()
                _write_roar(meta_dir / f"rating_{r}.roar", mask_ids)

    def dump_memmap(name: str, arr: np.ndarray, dtype) -> None:
        mm = np.memmap(meta_dir / f"{name}.bin", mode='w+', dtype=dtype, shape=arr.shape)
        mm[:] = arr.astype(dtype); mm.flush()

    dump_memmap("post_ids", post_ids, np.int64)
    dump_memmap("epoch_day", df.get_column("epoch_day").to_numpy(), np.int32)
    dump_memmap("w", df.get_column("w").to_numpy(), np.int32)
    dump_memmap("h", df.get_column("h").to_numpy(), np.int32)
    dump_memmap("score", df.get_column("score").to_numpy(), np.int32)
    dump_memmap("fav_count", df.get_column("fav_count").to_numpy(), np.int32)
    dump_memmap("is_deleted", df.get_column("is_deleted").cast(pl.UInt8).to_numpy(), np.uint8)
    dump_memmap("is_pending", df.get_column("is_pending").cast(pl.UInt8).to_numpy(), np.uint8)

    sentinel.write_text("ok")
    log(f"[mmaps] готово: {n} постов (post_ids отсортированы)")