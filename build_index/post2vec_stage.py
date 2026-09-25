from __future__ import annotations
"""
Post2Vec v2: compute post embeddings as categorially-weighted sums of Tag2Vec vectors.

Outputs:
  - features/post2vec.parquet: [post_id:int64, vec:list[float32]]
  - features/post2vec_sq8.index: 8-bit scalar quantized FAISS index (~755 MB for 5.9M posts)
  - features/post2vec_faiss.index: uncompressed FlatIP FAISS index
  - features/post2vec_faiss_ids.parquet: [row:int32, post_id:int64]
"""

from typing import Dict, List, Tuple

import numpy as np
import polars as pl
import scipy.sparse as sp  # type: ignore

try:
    import faiss  # type: ignore
except ImportError:
    faiss = None  # type: ignore

from .config import Config
from .utils import ensure_dir, newer_than, log


def _load_tag2vec(cfg: Config) -> Tuple[Dict[int, np.ndarray], int]:
    path = cfg.features_dir / "tag2vec.parquet"
    if not path.exists():
        raise FileNotFoundError(f"tag2vec not found: {path}")
    df = pl.read_parquet(path).select([pl.col("tag_id").cast(pl.Int64), "vec"])
    vecs: Dict[int, np.ndarray] = {}
    dim = 0
    for tag_id, vec in df.iter_rows():
        v = np.asarray(vec, dtype=np.float32)
        if dim == 0:
            dim = int(v.shape[0])
        vecs[int(tag_id)] = v
    return vecs, dim


def _load_tag_metadata(cfg: Config) -> Tuple[Dict[int, float], Dict[int, int]]:
    """Loads IDF and category for each tag_id from tags.parquet or tags_dict.parquet."""
    base = cfg.tags_parquet
    if base.exists():
        df = pl.read_parquet(base).select([
            pl.col("tag_id").cast(pl.Int64),
            pl.coalesce([pl.col("idf"), pl.lit(1.0)]).cast(pl.Float32).alias("idf"),
            pl.coalesce([pl.col("category"), pl.lit(0)]).cast(pl.Int32).alias("category"),
        ])
    else:
        dict_path = cfg.root / "tags_dict.parquet"
        if dict_path.exists():
            df = pl.read_parquet(dict_path).select([
                pl.col("tag_id").cast(pl.Int64),
                pl.lit(1.0, dtype=pl.Float32).alias("idf"),
                pl.coalesce([pl.col("category"), pl.lit(0)]).cast(pl.Int32).alias("category"),
            ])
        else:
            return {}, {}
    idf_map = {int(t): float(w) for t, w in zip(df["tag_id"], df["idf"])}
    cat_map = {int(t): int(c) for t, c in zip(df["tag_id"], df["category"])}
    return idf_map, cat_map


def _iter_post_tags(cfg: Config) -> pl.DataFrame:
    long = pl.scan_parquet(f"{cfg.post_tags_parquet.as_posix()}/**/*.parquet")
    posts = pl.scan_parquet(f"{cfg.posts_parquet.as_posix()}/**/*.parquet").select([
        pl.col("id").alias("post_id"),
        pl.col("is_deleted"),
        pl.col("is_pending"),
    ])
    if getattr(cfg, "reliable_only", True):
        posts = posts.filter(~pl.col("is_deleted") & ~pl.col("is_pending"))
    posts = posts.select(["post_id"])

    joined = long.join(posts, on="post_id", how="inner")
    grouped = (
        joined
        .with_columns(pl.col("tag_id").cast(pl.Int64))
        .group_by("post_id", maintain_order=True)
        .agg(pl.col("tag_id").unique().alias("tags"))
        .select([pl.col("post_id").cast(pl.Int64), pl.col("tags")])
    )
    return grouped.collect(engine="streaming")


def _prepare_weighted_tag_matrix(
    vecs: Dict[int, np.ndarray],
    idf: Dict[int, float],
    cat_map: Dict[int, int],
    cat_weights: Dict[int, float],
) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Constructs weighted tag matrix Ew where row i = w_cat(t) * idf(t) * e_t.
    Tags with category weight 0.0 (e.g. contributor, invalid, lore) are omitted.
    """
    tag_ids = sorted(vecs.keys())
    idx_of: Dict[int, int] = {}
    valid_tags: List[int] = []

    for t in tag_ids:
        cat = cat_map.get(t, 0)
        w_cat = cat_weights.get(cat, 1.0)
        if w_cat > 0.0:
            idx_of[t] = len(valid_tags)
            valid_tags.append(t)

    if not valid_tags:
        return np.zeros((0, 0), dtype=np.float32), idx_of

    D = int(next(iter(vecs.values())).shape[0])
    Ew = np.zeros((len(valid_tags), D), dtype=np.float32)
    for i, t in enumerate(valid_tags):
        cat = cat_map.get(t, 0)
        w_cat = float(cat_weights.get(cat, 1.0))
        w_idf = float(idf.get(t, 1.0))
        combined_weight = w_cat * w_idf
        if combined_weight != 0.0:
            Ew[i, :] = combined_weight * vecs[t]

    return Ew, idx_of


def _batch_compute_with_csr(
    rows: List[Tuple[int, List[int]]],
    idx_of: Dict[int, int],
    Ew: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Batch computes post embeddings via sparse matrix multiplication X = M @ Ew,
    followed by L2 normalization.
    """
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
        X = M @ Ew  # (B, T) @ (T, D) -> (B, D)

    # L2 normalize each row; if norm < 1e-6, zero it out
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    valid_mask = (norms > 1e-6)
    safe_norms = np.where(valid_mask, norms, 1.0)
    X = np.where(valid_mask, X / safe_norms, 0.0).astype(np.float32, copy=False)
    pids = np.asarray([int(pid) for pid, _ in rows], dtype=np.int64)
    return X, pids


def step_post2vec(cfg: Config) -> None:
    out = cfg.features_dir / "post2vec.parquet"
    idx_flat_path = cfg.features_dir / "post2vec_faiss.index"
    idx_sq8_path = cfg.features_dir / "post2vec_sq8.index"
    ids_path = cfg.features_dir / "post2vec_faiss_ids.parquet"

    in_vecs = cfg.features_dir / "tag2vec.parquet"
    in_posts = cfg.posts_parquet / "_SUCCESS"
    in_long = cfg.post_tags_parquet / "_SUCCESS"
    in_tags = cfg.tags_parquet

    if (not getattr(cfg, "force", False)) and out.exists() and newer_than(out, *[p for p in (in_vecs, in_posts, in_long, in_tags) if p.exists()]):
        log("[post2vec] already fresh - skip")
        return

    log("[post2vec] loading tag2vec, categories, and IDF...")
    tag_vecs, dim = _load_tag2vec(cfg)
    idf, cat_map = _load_tag_metadata(cfg)
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

    cat_weights = getattr(cfg, "category_weights", {
        1: 2.50, 4: 2.00, 3: 1.80, 5: 1.30, 0: 1.00, 7: 0.05, 2: 0.00, 6: 0.00, 8: 0.00
    })
    log(f"[post2vec] preparing weighted tag matrix with {len(cat_weights)} category weights...")
    Ew, idx_of = _prepare_weighted_tag_matrix(tag_vecs, idf, cat_map, cat_weights)

    log(f"[post2vec] computing vectors for {grouped.height:,} posts (dim={dim}, active_tags={len(idx_of):,})...")

    import pyarrow as pa
    import pyarrow.parquet as pq

    ensure_dir(cfg.features_dir)
    writer = None
    ids_writer = None
    row_offset = 0

    # Initialize FAISS indices if faiss is available
    faiss_flat = None
    faiss_sq8 = None
    sq8_trained = False

    if faiss is not None:
        faiss_flat = faiss.IndexFlatIP(dim)
        if getattr(cfg, "post2vec_index_sq8", True):
            faiss_sq8 = faiss.IndexScalarQuantizer(dim, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT)

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
                X_contig = np.ascontiguousarray(X.astype(np.float32, copy=False))

                if faiss_sq8 is not None:
                    if not sq8_trained:
                        train_sample = X_contig[:min(len(X_contig), 50_000)]
                        faiss_sq8.train(train_sample)
                        sq8_trained = True
                    faiss_sq8.add(X_contig)

                if faiss_flat is not None:
                    faiss_flat.add(X_contig)

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

    log(f"[post2vec] saved {row_offset:,} vectors to {out}")

    # Save FAISS indices
    if faiss is not None and row_offset > 0:
        if faiss_flat is not None:
            try:
                faiss.write_index(faiss_flat, idx_flat_path.as_posix())
                log(f"[post2vec] Flat FAISS index saved: {idx_flat_path.as_posix()}")
            except Exception as e:
                log(f"[post2vec] Flat FAISS write failed: {e}")

        if faiss_sq8 is not None:
            try:
                faiss.write_index(faiss_sq8, idx_sq8_path.as_posix())
                log(f"[post2vec] SQ8 FAISS index saved: {idx_sq8_path.as_posix()}")
            except Exception as e:
                log(f"[post2vec] SQ8 FAISS write failed: {e}")

    log("[post2vec] done")
