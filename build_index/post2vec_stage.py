from __future__ import annotations
"""
Post2Vec: compute post embeddings as weighted sums of Tag2Vec vectors.

Outputs:
  - features/post2vec.parquet: [post_id:int64, vec:list[float32]]
  - features/post2vec_faiss.index
  - features/post2vec_faiss_ids.parquet: [row:int32, post_id:int64]
"""

from typing import Dict, List, Tuple

import numpy as np
import polars as pl
import scipy.sparse as sp  # type: ignore
import faiss  # type: ignore

from .config import Config
from .utils import ensure_dir, newer_than, log


def _load_tag2vec(cfg: Config) -> Tuple[Dict[int, np.ndarray], int]:
    path = cfg.features_dir / "tag2vec.parquet"
    if not path.exists():
        raise FileNotFoundError(f"tag2vec not found: {path}")
    df = pl.read_parquet(path).select([pl.col("tag_id").cast(pl.Int32), "vec"])  # tag_id:int32, vec:list[f32]
    vecs: Dict[int, np.ndarray] = {}
    dim = 0
    for tag_id, vec in df.iter_rows():
        v = np.asarray(vec, dtype=np.float32)
        if dim == 0:
            dim = int(v.shape[0])
        vecs[int(tag_id)] = v
    return vecs, dim


def _load_idf(cfg: Config) -> Dict[int, float]:
    base = cfg.tags_parquet
    if base.exists():
        df = pl.read_parquet(base).select([
            pl.col("tag_id").cast(pl.Int32),
            pl.coalesce([pl.col("idf"), pl.lit(1.0)]).alias("idf"),
        ])
    else:
        df = pl.read_parquet(cfg.root / "tags_dict.parquet").select([
            pl.col("tag_id").cast(pl.Int32),
        ]).with_columns(pl.lit(1.0).alias("idf"))
    return {int(t): float(w) for t, w in df.iter_rows()}


def _iter_post_tags(cfg: Config) -> pl.DataFrame:
    long = pl.scan_parquet(f"{cfg.post_tags_parquet.as_posix()}/**/*.parquet")
    posts = pl.scan_parquet(f"{cfg.posts_parquet.as_posix()}/**/*.parquet").select([
        pl.col("id").alias("post_id"),
        pl.col("is_deleted"),
        pl.col("is_pending"),
    ])
    if getattr(cfg, "reliable_only", True):
        posts = posts.filter(~pl.col("is_deleted") & ~pl.col("is_pending"))
    posts = posts.select(["post_id"])  # ids only

    joined = long.join(posts, on="post_id", how="inner")
    grouped = (
        joined
        .with_columns(pl.col("tag_id").cast(pl.Int32))
        .group_by("post_id", maintain_order=True)
        .agg(pl.col("tag_id").unique().alias("tags"))
        .select([pl.col("post_id").cast(pl.Int64), pl.col("tags")])
    )
    return grouped.collect(engine="streaming")


def _prepare_weighted_tag_matrix(vecs: Dict[int, np.ndarray], idf: Dict[int, float]) -> Tuple[np.ndarray, Dict[int, int]]:
    tag_ids = sorted(vecs.keys())
    idx_of: Dict[int, int] = {t: i for i, t in enumerate(tag_ids)}
    if not tag_ids:
        return np.zeros((0, 0), dtype=np.float32), idx_of
    D = int(next(iter(vecs.values())).shape[0])
    Ew = np.zeros((len(tag_ids), D), dtype=np.float32)
    for i, t in enumerate(tag_ids):
        w = float(idf.get(t, 1.0))
        if w != 0.0:
            Ew[i, :] = w * vecs[t]
    return Ew, idx_of


def _batch_compute_with_csr(rows: List[Tuple[int, List[int]]], idx_of: Dict[int, int], Ew: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    B = len(rows)
    if B == 0:
        return np.zeros((0, Ew.shape[1]), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    indptr = [0]
    indices: List[int] = []
    for _, tags in rows:
        for t in tags:
            j = idx_of.get(int(t))
            if j is not None:
                indices.append(j)
        indptr.append(len(indices))
    data = np.ones((len(indices),), dtype=np.float32)
    indptr_arr = np.asarray(indptr, dtype=np.int64)
    indices_arr = np.asarray(indices, dtype=np.int32)

    X = np.zeros((B, Ew.shape[1]), dtype=np.float32)
    if Ew.size > 0 and indices_arr.size > 0:
        M = sp.csr_matrix((data, indices_arr, indptr_arr), shape=(B, Ew.shape[0]), dtype=np.float32)
        X = M @ Ew  # (B,T) @ (T,D) -> (B,D)
    n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    X = np.divide(X, n, out=np.zeros_like(X), where=(n > 0))
    pids = np.asarray([int(pid) for pid, _ in rows], dtype=np.int64)
    return X, pids


def step_post2vec(cfg: Config) -> None:
    out = cfg.features_dir / "post2vec.parquet"

    in_vecs = cfg.features_dir / "tag2vec.parquet"
    in_posts = cfg.posts_parquet / "_SUCCESS"
    in_long = cfg.post_tags_parquet / "_SUCCESS"
    in_tags = cfg.tags_parquet

    if (not getattr(cfg, "force", False)) and out.exists() and newer_than(out, *[p for p in (in_vecs, in_posts, in_long, in_tags) if p.exists()]):
        log("[post2vec] already fresh - skip")
        return

    log("[post2vec] loading tag2vec and IDF...")
    tag_vecs, dim = _load_tag2vec(cfg)
    idf = _load_idf(cfg)
    if dim <= 0:
        ensure_dir(cfg.features_dir)
        pl.DataFrame({"post_id": [], "vec": []}).write_parquet(out, compression="zstd")
        log("[post2vec] empty tag vectors; wrote empty output")
        return

    log("[post2vec] aggregating tags per post...")
    grouped = _iter_post_tags(cfg)
    if grouped.is_empty():
        ensure_dir(cfg.features_dir)
        pl.DataFrame({"post_id": [], "vec": []}).write_parquet(out, compression="zstd")
        log("[post2vec] no posts; wrote empty output")
        return

    log(f"[post2vec] computing vectors for {grouped.height:,} posts (dim={dim})...")
    Ew, idx_of = _prepare_weighted_tag_matrix(tag_vecs, idf)

    import pyarrow as pa
    import pyarrow.parquet as pq

    ensure_dir(cfg.features_dir)
    writer = None
    # Prepare FAISS index and ids writer to build incrementally
    idx_path = cfg.features_dir / "post2vec_faiss.index"
    ids_path = cfg.features_dir / "post2vec_faiss_ids.parquet"
    faiss_index = faiss.IndexFlatIP(dim)
    ids_writer = None
    row_offset = 0
    try:
        chunk = int(getattr(cfg, "post2vec_batch", 200_000))
        for i in range(0, grouped.height, chunk):
            part = grouped.slice(i, min(chunk, grouped.height - i))
            rows: List[Tuple[int, List[int]]] = [(int(pid), list(tags)) for pid, tags in part.iter_rows()]
            X, pids = _batch_compute_with_csr(rows, idx_of, Ew)

            arr_ids = pa.array(pids.tolist(), type=pa.int64())
            arr_vec = pa.FixedSizeListArray.from_arrays(pa.array(X.reshape(-1), type=pa.float32()), dim)
            batch_tbl = pa.table({"post_id": arr_ids, "vec": arr_vec})

            if writer is None:
                writer = pq.ParquetWriter(out.as_posix(), batch_tbl.schema, compression="zstd")
            writer.write_table(batch_tbl)

            if X.size > 0:
                faiss_index.add(np.ascontiguousarray(X.astype(np.float32, copy=False)))
                arr_row = pa.array(np.arange(row_offset, row_offset + pids.shape[0], dtype=np.int32))
                ids_tbl = pa.table({"row": arr_row, "post_id": arr_ids})
                if ids_writer is None:
                    ids_writer = pq.ParquetWriter(ids_path.as_posix(), ids_tbl.schema, compression="zstd")
                ids_writer.write_table(ids_tbl)
                row_offset += pids.shape[0]
    finally:
        if writer is not None:
            writer.close()
        if ids_writer is not None:
            ids_writer.close()

    log(f"[post2vec] saved vectors to {out}")

    try:
        if row_offset > 0:
            faiss.write_index(faiss_index, idx_path.as_posix())
            log(f"[post2vec] FAISS index saved: {idx_path.as_posix()} ; ids: {ids_path.as_posix()}")
    except Exception as e:
        log(f"[post2vec] FAISS index build failed: {e}")

    log("[post2vec] done")

