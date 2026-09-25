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
    post_ids_bin = cfg.mmaps_dir / "post_ids.bin"
    mmaps_sentinel = cfg.mmaps_dir / "_SUCCESS"

    if not post_ids_bin.exists():
        raise FileNotFoundError(
            f"Required primary key array {post_ids_bin} does not exist. "
            "The 'mmaps' stage must be executed before 'bitmaps' to establish the ID bijection."
        )

    if (
        sentinel.exists()
        and not cfg.force
        and newer_than(sentinel, cfg.post_tags_parquet / "_SUCCESS", mmaps_sentinel)
    ):
        log("[bitmaps] already fresh - skip")
        return

    log("[bitmaps] building roaring-indexes with dense ID bijection...")
    global_post_ids = np.memmap(post_ids_bin, mode="r", dtype=np.int64)
    n_posts = len(global_post_ids)

    shard_dirs = sorted(p for p in cfg.post_tags_parquet.glob("tag_shard=*") if p.is_dir())
    legacy_tag_dirs = sorted(p for p in cfg.post_tags_parquet.glob("tag_id=*") if p.is_dir())

    if shard_dirs:
        # Sharded mode: tag_shard=*
        def process_shard(shard_dir: Path) -> int:
            tbl = pl.scan_parquet(f"{shard_dir.as_posix()}/**/*.parquet").select(["post_id", "tag_id"]).collect()
            if tbl.is_empty():
                return 0

            grouped = tbl.group_by("tag_id").agg(pl.col("post_id")).sort("tag_id")
            shard_id = int(shard_dir.name.split("=")[1])

            pairs: List[Tuple[int, np.ndarray]] = []
            for tag_id, post_ids in grouped.iter_rows():
                raw_ids = np.array(post_ids, dtype=np.int64)
                raw_ids.sort()
                # Bijection: post_id(int64) -> dense_id(uint32) in [0, N-1]
                dense_idx = np.searchsorted(global_post_ids, raw_ids)
                in_bounds = dense_idx < n_posts
                safe_idx = np.where(in_bounds, dense_idx, 0)
                valid = in_bounds & (global_post_ids[safe_idx] == raw_ids)
                dense_ids = dense_idx[valid].astype(np.uint32)
                pairs.append((int(tag_id), dense_ids))

            out_pack = cfg.bitmaps_dir / f"shard_{shard_id:04d}.roarpack"
            out_index = cfg.bitmaps_dir / f"index_{shard_id:04d}.bin"
            _write_roar_shard(out_pack, out_index, pairs)
            return grouped.height

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=cfg.workers) as ex:
            list(tqdm(ex.map(process_shard, shard_dirs), total=len(shard_dirs), desc="bitmaps"))

    else:
        # Legacy-mode: tag_id=*
        tag_parts = legacy_tag_dirs

        def process_part(part_dir: Path) -> Tuple[int, np.ndarray]:
            tag_id = int(part_dir.name.split("=")[1])
            tbl = pl.scan_parquet(f"{part_dir.as_posix()}/*.parquet").collect()
            raw_ids = tbl.get_column("post_id").to_numpy().astype(np.int64)
            raw_ids.sort()
            dense_idx = np.searchsorted(global_post_ids, raw_ids)
            in_bounds = dense_idx < n_posts
            safe_idx = np.where(in_bounds, dense_idx, 0)
            valid = in_bounds & (global_post_ids[safe_idx] == raw_ids)
            dense_ids = dense_idx[valid].astype(np.uint32)
            return (tag_id, dense_ids)

        acc: List[Tuple[int, np.ndarray]] = []
        for part_dir in tqdm(tag_parts, desc="collect"):
            tag_id, ids = process_part(part_dir)
            if ids is not None and len(ids) > 0:
                acc.append((tag_id, ids))

        acc.sort(key=lambda t: t[0])
        shard = cfg.roar_shard_size if cfg.roar_shard_size > 0 else 10000
        for i in tqdm(range(0, len(acc), shard), desc="shard"):
            chunk = acc[i:i + shard]
            if not chunk:
                continue
            shard_id = i // shard
            out_pack = cfg.bitmaps_dir / f"shard_{shard_id:04d}.roarpack"
            out_index = cfg.bitmaps_dir / f"index_{shard_id:04d}.bin"
            pairs = [(int(tag_id), ids) for tag_id, ids in chunk]
            _write_roar_shard(out_pack, out_index, pairs)

    sentinel.write_text("ok")
    log("[bitmaps] done")


def step_build_mmaps(cfg: Config) -> None:
    meta_dir = cfg.mmaps_dir
    ensure_dir(meta_dir)
    sentinel = meta_dir / "_SUCCESS"
    if sentinel.exists() and not cfg.force and newer_than(sentinel, cfg.posts_parquet / "_SUCCESS"):
        log("[mmaps] already fresh - skip")
        return

    log("[mmaps] collect compact columns...")
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
            pl.col("file_ext"),
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
            pl.col("file_ext"),
        ]).collect(engine="streaming")
        df = df.with_columns(pl.lit(None).alias("rating"))

    # Strictly sort by id to establish the canonical bijection A = [p_0, ..., p_{N-1}]
    df = df.sort("id")
    post_ids = df.get_column("id").to_numpy().astype(np.int64)
    n = post_ids.size

    # Build rating roars using dense uint32 indices in [0, N-1]
    rating_col = df.get_column("rating").to_numpy()
    for r in ("s", "q", "e"):
        dense_mask = np.where(rating_col == r)[0].astype(np.uint32)
        _write_roar(meta_dir / f"rating_{r}.roar", dense_mask)

    def dump_memmap(name: str, arr: np.ndarray, dtype) -> None:
        mm = np.memmap(meta_dir / f"{name}.bin", mode='w+', dtype=dtype, shape=arr.shape)
        mm[:] = arr.astype(dtype)
        mm.flush()

    dump_memmap("post_ids", post_ids, np.int64)
    # Map rating strings to uint8 codes: s=0, q=1, e=2
    rating_code_map = {"s": 0, "q": 1, "e": 2}
    rating_codes = np.array([rating_code_map.get(str(r).lower(), 0) for r in rating_col], dtype=np.uint8)
    dump_memmap("rating", rating_codes, np.uint8)
    dump_memmap("epoch_day", df.get_column("epoch_day").to_numpy(), np.int32)
    dump_memmap("w", df.get_column("w").to_numpy(), np.int32)
    dump_memmap("h", df.get_column("h").to_numpy(), np.int32)
    dump_memmap("score", df.get_column("score").to_numpy(), np.int32)
    dump_memmap("fav_count", df.get_column("fav_count").to_numpy(), np.int32)
    dump_memmap("is_deleted", df.get_column("is_deleted").cast(pl.UInt8).to_numpy(), np.uint8)
    dump_memmap("is_pending", df.get_column("is_pending").cast(pl.UInt8).to_numpy(), np.uint8)

    # Media file_ext:
    # 0: png, 1: jpg, 2: webp  (IMAGES)
    # 3: gif, 4: webm, 5: mp4  (VIDEOS & ANIMATIONS)
    # 6: swf, 7: other         (BANNED)
    ext_code_map = {"png": 0, "jpg": 1, "jpeg": 1, "webp": 2, "gif": 3, "webm": 4, "mp4": 5, "swf": 6}
    ext_strings = [str(x).lower().strip() if x is not None else "" for x in df.get_column("file_ext").to_list()]
    ext_codes = np.array([ext_code_map.get(s, 7) for s in ext_strings], dtype=np.uint8)
    dump_memmap("file_ext", ext_codes, np.uint8)

    is_del_col = df.get_column("is_deleted").to_numpy().astype(bool)
    _write_roar(meta_dir / "not_deleted.roar", np.where((~is_del_col) & (ext_codes < 6))[0].astype(np.uint32))
    _write_roar(meta_dir / "media_image.roar", np.where((~is_del_col) & (ext_codes <= 2))[0].astype(np.uint32))
    _write_roar(meta_dir / "media_video.roar", np.where((~is_del_col) & (ext_codes >= 3) & (ext_codes <= 5))[0].astype(np.uint32))

    sentinel.write_text("ok")
    log(f"[mmaps] done: {n:,} posts (post_ids sorted, rating & media roars generated)")