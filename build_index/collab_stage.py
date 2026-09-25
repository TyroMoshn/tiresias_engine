from __future__ import annotations
"""
Collaborative Filtering & Taste Personas:
- Parse user_favorites.csv into post co-favorited graph (post_cofav.parquet)
- Compute user taste vectors and cluster into 64 latent taste archetypes (taste_archetypes.parquet + taste_centroids.npy)
"""

import gc
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.sparse as sp  # type: ignore

try:
    import faiss  # type: ignore
except ImportError:
    faiss = None  # type: ignore

from .config import Config
from .utils import ensure_dir, newer_than, log


def _parse_post_ids_field(s: str) -> List[int]:
    """Parse postgres-style int array string '{1,2,3}' into a list of ints."""
    if not s or s == "{}":
        return []
    s = s.strip()
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1]
    if not s:
        return []
    out: List[int] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(int(tok))
        except ValueError:
            continue
    return sorted(set(out))


def step_collab_favs(cfg: Config) -> None:
    """
    Build post co-favorited graph from user_favorites.csv.
    Each user's contribution to pair (a, b) is weighted by 1 / sqrt(|F_u|).
    Uses chunked row-slice projection to maintain memory < 500 MB even for 3M+ posts.
    Saves: features/post_cofav.parquet [a_post_id: int64, b_post_id: int64, weight: float32]
    """
    out = cfg.post_cofav_parquet
    fav_csv = cfg.user_favorites_csv

    if not fav_csv.exists():
        log(f"[collab] '{fav_csv.name}' not found at {fav_csv} - skipping collab graph.")
        return

    if out.exists() and not cfg.force and newer_than(out, fav_csv):
        log("[collab] post_cofav.parquet already fresh - skip")
        return

    ensure_dir(out.parent)
    log(f"[collab] parsing {fav_csv.name} for co-favorited graph...")

    # Read user favorites
    df = (
        pl.read_csv(fav_csv, infer_schema_length=0, ignore_errors=True)
        .select([
            pl.col("user_id").cast(pl.Int64, strict=False),
            pl.col("favorites").cast(pl.Utf8, strict=False),
        ])
        .drop_nulls()
    )

    user_rows: List[Tuple[int, List[int]]] = []
    unique_pids: Set[int] = set()

    for u_id, fav_str in df.iter_rows():
        pids = _parse_post_ids_field(fav_str)
        if len(pids) < cfg.fav_user_min_posts:
            continue
        # Truncate mega-collections to prevent combinatorial explosion
        if len(pids) > cfg.fav_user_max_posts:
            pids = pids[:cfg.fav_user_max_posts]
        user_rows.append((int(u_id), pids))
        unique_pids.update(pids)

    n_users = len(user_rows)
    n_unique_posts = len(unique_pids)
    log(f"[collab] {n_users:,} valid users, {n_unique_posts:,} unique favorited posts")

    schema = pa.schema([
        ("a_post_id", pa.int64()),
        ("b_post_id", pa.int64()),
        ("weight", pa.float32()),
    ])

    if n_users == 0 or n_unique_posts == 0:
        empty_tbl = pa.Table.from_arrays([
            pa.array([], type=pa.int64()),
            pa.array([], type=pa.int64()),
            pa.array([], type=pa.float32()),
        ], schema=schema)
        pq.write_table(empty_tbl, out.as_posix(), compression="zstd")
        log("[collab] no valid favorites data; wrote empty cofav output")
        return

    # Map post_ids to contiguous dense indices [0, P-1]
    sorted_pids = sorted(unique_pids)
    pid_to_idx: Dict[int, int] = {pid: i for i, pid in enumerate(sorted_pids)}
    idx_to_pid = np.array(sorted_pids, dtype=np.int64)

    # Build sparse user-post matrix M where M[u, p] = 1 / (|F_u| ** 0.25)
    # Then (M.T @ M)[a, b] = sum_u (1 / sqrt(|F_u|)) = W_cofav(a, b)
    row_ind: List[int] = []
    col_ind: List[int] = []
    data_vals: List[float] = []

    for u_idx, (_, pids) in enumerate(user_rows):
        w_u = float(1.0 / (len(pids) ** 0.25))
        for pid in pids:
            col = pid_to_idx.get(pid)
            if col is not None:
                row_ind.append(u_idx)
                col_ind.append(col)
                data_vals.append(w_u)

    del user_rows
    del unique_pids
    gc.collect()

    M = sp.csr_matrix(
        (data_vals, (row_ind, col_ind)),
        shape=(n_users, n_unique_posts),
        dtype=np.float32
    )
    del row_ind, col_ind, data_vals
    gc.collect()

    Mt = M.T.tocsr()

    log(f"[collab] streaming cofav edges using chunked projection across {n_unique_posts:,} posts...")

    min_w = float(cfg.cofav_min_weight)
    max_edges = int(cfg.cofav_max_edges_per_post)
    chunk_size = 5000

    writer = pq.ParquetWriter(out.as_posix(), schema=schema, compression="zstd")
    total_edges = 0

    accum_a: List[int] = []
    accum_b: List[int] = []
    accum_w: List[float] = []

    for p_start in range(0, n_unique_posts, chunk_size):
        p_end = min(p_start + chunk_size, n_unique_posts)
        Mt_chunk = Mt[p_start:p_end, :]
        if Mt_chunk.nnz == 0:
            continue

        # C_chunk shape: (chunk_size, n_unique_posts)
        C_chunk = (Mt_chunk @ M).tocsr()
        indptr = C_chunk.indptr
        indices = C_chunk.indices
        c_data = C_chunk.data

        for r in range(p_end - p_start):
            g_a = p_start + r
            start_idx = indptr[r]
            end_idx = indptr[r + 1]
            if start_idx == end_idx:
                continue

            cols = indices[start_idx:end_idx]
            vals = c_data[start_idx:end_idx]

            # Keep strictly upper triangle (b > a) and weight >= min_w
            mask = (cols > g_a) & (vals >= min_w)
            if not np.any(mask):
                continue

            v_cols = cols[mask]
            v_vals = vals[mask]

            if max_edges > 0 and len(v_cols) > max_edges:
                top_idx = np.argpartition(-v_vals, max_edges)[:max_edges]
                order = np.argsort(-v_vals[top_idx])
                top_idx = top_idx[order]
                v_cols = v_cols[top_idx]
                v_vals = v_vals[top_idx]
            else:
                order = np.argsort(-v_vals)
                v_cols = v_cols[order]
                v_vals = v_vals[order]

            real_a = int(idx_to_pid[g_a])
            for col_idx, weight_val in zip(v_cols, v_vals):
                accum_a.append(real_a)
                accum_b.append(int(idx_to_pid[col_idx]))
                accum_w.append(float(weight_val))

        del C_chunk

        if len(accum_a) >= 200_000:
            tbl = pa.Table.from_arrays([
                pa.array(accum_a, type=pa.int64()),
                pa.array(accum_b, type=pa.int64()),
                pa.array(accum_w, type=pa.float32()),
            ], schema=schema)
            writer.write_table(tbl)
            total_edges += len(accum_a)
            accum_a.clear()
            accum_b.clear()
            accum_w.clear()

    # Flush remaining edges
    if len(accum_a) > 0:
        tbl = pa.Table.from_arrays([
            pa.array(accum_a, type=pa.int64()),
            pa.array(accum_b, type=pa.int64()),
            pa.array(accum_w, type=pa.float32()),
        ], schema=schema)
        writer.write_table(tbl)
        total_edges += len(accum_a)
        accum_a.clear()
        accum_b.clear()
        accum_w.clear()

    writer.close()
    del M, Mt, pid_to_idx, idx_to_pid
    gc.collect()

    log(f"[collab] post_cofav.parquet written: {total_edges:,} edges")


def step_taste_archetypes(cfg: Config) -> None:
    """
    Cluster user preference vectors into K=64 taste archetypes.
    Generates:
      - features/taste_archetypes.parquet: [archetype_id: int32, post_id: int64, rank: int32, score: float32]
      - features/taste_centroids.npy: (K, dim) normalized centroids
    """
    from sklearn.cluster import MiniBatchKMeans

    out_archetypes = cfg.taste_archetypes_parquet
    out_centroids = cfg.taste_centroids_npy
    fav_csv = cfg.user_favorites_csv
    vec_path = cfg.features_dir / "post2vec.parquet"
    sq8_idx_path = cfg.features_dir / "post2vec_sq8.index"
    flat_idx_path = cfg.features_dir / "post2vec_faiss.index"
    ids_path = cfg.features_dir / "post2vec_faiss_ids.parquet"

    if not fav_csv.exists():
        log(f"[collab] '{fav_csv.name}' not found at {fav_csv} - skipping taste archetypes.")
        return

    if not vec_path.exists():
        log(f"[collab] '{vec_path.name}' not found - run post2vec first.")
        return

    if (
        out_archetypes.exists()
        and out_centroids.exists()
        and not cfg.force
        and newer_than(out_archetypes, fav_csv, vec_path)
    ):
        log("[collab] taste archetypes already fresh - skip")
        return

    ensure_dir(out_archetypes.parent)
    log(f"[collab] extracting user favorite lists for taste persona clustering...")

    # 1. Parse user favorites
    df = (
        pl.read_csv(fav_csv, infer_schema_length=0, ignore_errors=True)
        .select([
            pl.col("user_id").cast(pl.Int64, strict=False),
            pl.col("favorites").cast(pl.Utf8, strict=False),
        ])
        .drop_nulls()
    )

    valid_users: List[int] = []
    post_to_users: Dict[int, List[int]] = {}

    for u_id, fav_str in df.iter_rows():
        pids = _parse_post_ids_field(fav_str)
        if len(pids) >= cfg.fav_user_min_posts:
            if len(pids) > cfg.fav_user_max_posts:
                pids = pids[:cfg.fav_user_max_posts]
            u_idx = len(valid_users)
            valid_users.append(int(u_id))
            for pid in pids:
                if pid not in post_to_users:
                    post_to_users[pid] = [u_idx]
                else:
                    post_to_users[pid].append(u_idx)

    n_users = len(valid_users)
    log(f"[collab] {n_users:,} valid users referencing {len(post_to_users):,} unique posts")

    if n_users < cfg.taste_archetypes_k:
        log(f"[collab] not enough valid users ({n_users}) for K={cfg.taste_archetypes_k} clusters")
        return

    # 2. Inspect post2vec.parquet to determine vector dimension
    pf = pq.ParquetFile(vec_path.as_posix())
    dim = 0
    if pf.metadata.num_rows > 0:
        for b in pf.iter_batches(batch_size=1, columns=["post_id", "vec"]):
            vecs = b["vec"].to_pylist()
            if vecs and vecs[0] is not None:
                dim = len(vecs[0])
            break

    if dim == 0:
        log("[collab] empty post vectors; skipping taste archetypes.")
        return

    # 3. Stream through post2vec.parquet and accumulate user vectors
    # TODO (Future Optimization): To accelerate the ~7-minute streaming pass over 5.16M posts,
    # convert the FixedSizeListArray directly into a contiguous 2D NumPy array block per batch
    # instead of Python to_pylist(), allowing vectorized index scattering.
    log(f"[collab] streaming post2vec.parquet to accumulate user embeddings (dim={dim})...")
    user_sums = np.zeros((n_users, dim), dtype=np.float32)
    user_counts = np.zeros(n_users, dtype=np.int32)

    # In case FAISS is not available, we can also keep a sample of post vectors for fallback
    fallback_sample_posts: List[int] = []
    fallback_sample_vecs: List[np.ndarray] = []
    need_fallback_sample = (faiss is None) or (not sq8_idx_path.exists() and not flat_idx_path.exists())

    for batch in pf.iter_batches(batch_size=50_000, columns=["post_id", "vec"]):
        pids_arr = batch["post_id"].to_numpy(zero_copy_only=False)
        vecs_raw = batch["vec"].to_pylist()
        for pid, raw_vec in zip(pids_arr, vecs_raw):
            if raw_vec is None:
                continue
            int_pid = int(pid)
            u_list = post_to_users.get(int_pid)
            if u_list is not None:
                v = np.asarray(raw_vec, dtype=np.float32)
                for u_idx in u_list:
                    user_sums[u_idx] += v
                    user_counts[u_idx] += 1
                if need_fallback_sample and len(fallback_sample_posts) < 50_000:
                    fallback_sample_posts.append(int_pid)
                    fallback_sample_vecs.append(v)

    del post_to_users
    gc.collect()

    # 4. Filter and normalize user taste vectors
    valid_mask = user_counts >= cfg.fav_user_min_posts
    n_active = int(np.sum(valid_mask))
    log(f"[collab] {n_active:,} users with >= {cfg.fav_user_min_posts} favorited post vectors")

    if n_active < cfg.taste_archetypes_k:
        log(f"[collab] not enough active users ({n_active}) for K={cfg.taste_archetypes_k} clusters")
        return

    U_active = user_sums[valid_mask]
    norms = np.linalg.norm(U_active, axis=1, keepdims=True)
    valid_norm_mask = (norms[:, 0] > 1e-6)
    U_active = U_active[valid_norm_mask]
    norms = norms[valid_norm_mask]
    U_matrix = (U_active / norms).astype(np.float32, copy=False)

    del user_sums, user_counts, U_active
    gc.collect()

    # 5. Spherical K-Means clustering (MiniBatchKMeans with unit sphere projection)
    K = int(cfg.taste_archetypes_k)
    log(f"[collab] clustering into K={K} taste archetypes using MiniBatchKMeans...")
    mb_batch = max(3072, min(8192, len(U_matrix))) if len(U_matrix) >= 3072 else len(U_matrix)
    kmeans = MiniBatchKMeans(
        n_clusters=K,
        batch_size=mb_batch,
        random_state=42,
        n_init=3,
    )
    kmeans.fit(U_matrix)
    centroids = kmeans.cluster_centers_.astype(np.float32, copy=False)
    # Project centroids to unit sphere S^{D-1}
    c_norms = np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12
    centroids = (centroids / c_norms).astype(np.float32, copy=False)

    # Save centroids
    np.save(out_centroids, centroids)
    log(f"[collab] saved {K} normalized centroids to {out_centroids}")

    # 6. Retrieve top posts for each archetype via FAISS index (or fallback)
    faiss_index = None
    if faiss is not None:
        if sq8_idx_path.exists():
            faiss_index = faiss.read_index(sq8_idx_path.as_posix())
            log(f"[collab] using SQ8 index for archetype post retrieval ({sq8_idx_path.name})")
        elif flat_idx_path.exists():
            faiss_index = faiss.read_index(flat_idx_path.as_posix())
            log(f"[collab] using Flat index for archetype post retrieval ({flat_idx_path.name})")

    top_m = int(cfg.taste_archetypes_top_posts)
    archetype_rows: List[Tuple[int, int, int, float]] = []

    if faiss_index is not None and ids_path.exists():
        ids_df = pl.read_parquet(ids_path)
        row_to_pid = ids_df["post_id"].to_numpy()

        distances, rows = faiss_index.search(centroids, top_m)
        for arch_id in range(K):
            rank = 1
            for r_idx, score in zip(rows[arch_id], distances[arch_id]):
                if r_idx < 0 or r_idx >= len(row_to_pid):
                    continue
                pid = int(row_to_pid[r_idx])
                archetype_rows.append((arch_id, pid, rank, float(score)))
                rank += 1
    else:
        log("[collab] FAISS index not found - searching against sample posts...")
        if fallback_sample_vecs:
            V_sample = np.asarray(fallback_sample_vecs, dtype=np.float32)
            sim_matrix = centroids @ V_sample.T
            for arch_id in range(K):
                actual_m = min(top_m, len(fallback_sample_posts))
                top_indices = np.argsort(-sim_matrix[arch_id])[:actual_m]
                for rank, idx in enumerate(top_indices, start=1):
                    pid = fallback_sample_posts[idx]
                    score = float(sim_matrix[arch_id, idx])
                    archetype_rows.append((arch_id, pid, rank, score))

    res_df = pl.DataFrame(
        archetype_rows,
        schema={
            "archetype_id": pl.Int32,
            "post_id": pl.Int64,
            "rank": pl.Int32,
            "score": pl.Float32,
        },
        orient="row"
    )
    res_df.write_parquet(out_archetypes, compression="zstd")
    log(f"[collab] taste_archetypes.parquet written: {res_df.height:,} rows for {K} archetypes")
    log("[collab] done")
