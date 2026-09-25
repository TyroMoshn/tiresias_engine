from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl

try:
    import faiss  # type: ignore
except ImportError:
    faiss = None  # type: ignore


class FaissWrapper:
    """
    Manages the 8-bit scalar quantized FAISS index (post2vec_sq8.index).
    Provides sub-10ms nearest-neighbor semantic search over 5M+ posts.
    """

    def __init__(self, index_path: Path, ids_path: Path, num_threads: int = 0) -> None:
        self.index_path = index_path
        self.ids_path = ids_path
        self.num_threads = num_threads
        self.index = None
        self.row_to_pid: Optional[np.ndarray] = None
        self.pid_to_row: Dict[int, int] = {}
        self.sorted_pids: Optional[np.ndarray] = None
        self.sorted_idx: Optional[np.ndarray] = None
        self.total_vectors: int = 0
        self.dim: int = 128

        self._load()

    def _load(self) -> None:
        # 1. Load row -> post_id mapping
        if self.ids_path.exists():
            try:
                ids_df = pl.read_parquet(self.ids_path.as_posix())
                self.row_to_pid = ids_df["post_id"].to_numpy().astype(np.int64, copy=False)
                self.total_vectors = len(self.row_to_pid)
            except Exception:
                pass

        # 2. Load FAISS index
        if faiss is not None and self.index_path.exists():
            try:
                self.index = faiss.read_index(self.index_path.as_posix())
                self.dim = int(self.index.d)
                try:
                    threads = self.num_threads if self.num_threads > 0 else (os.cpu_count() or 4)
                    faiss.omp_set_num_threads(threads)
                except Exception:
                    pass
            except Exception:
                self.index = None

    def is_ready(self) -> bool:
        return self.index is not None and self.row_to_pid is not None

    def search_vector(self, query_vec: np.ndarray, top_k: int = 300) -> Tuple[List[int], List[float]]:
        """
        Performs inner product / cosine similarity search on the SQ8 index.
        Returns (post_ids, similarities).
        """
        if not self.is_ready():
            return [], []

        q = query_vec.reshape(1, -1).astype(np.float32, copy=False)
        norm = float(np.linalg.norm(q))
        if norm > 1e-6:
            q = q / norm

        distances, rows = self.index.search(q, top_k)
        result_pids: List[int] = []
        result_sims: List[float] = []

        for r_idx, sim in zip(rows[0], distances[0]):
            if r_idx < 0 or r_idx >= len(self.row_to_pid):
                continue
            pid = int(self.row_to_pid[r_idx])
            result_pids.append(pid)
            result_sims.append(float(sim))

        return result_pids, result_sims

    def _ensure_search_index(self) -> None:
        if self.sorted_pids is None and self.row_to_pid is not None and len(self.row_to_pid) > 0:
            idx = np.argsort(self.row_to_pid).astype(np.int32)
            self.sorted_pids = self.row_to_pid[idx]
            self.sorted_idx = idx

    def find_row_by_post_id(self, post_id: int) -> Optional[int]:
        """O(log N) lookup of row index in FAISS given 64-bit post ID."""
        if not self.is_ready() or self.row_to_pid is None:
            return None
        self._ensure_search_index()
        if self.sorted_pids is None or self.sorted_idx is None:
            return None
        pos = int(np.searchsorted(self.sorted_pids, post_id))
        if pos < len(self.sorted_pids) and self.sorted_pids[pos] == post_id:
            return int(self.sorted_idx[pos])
        return None

    def reconstruct_post_vector(self, post_id: int) -> Optional[np.ndarray]:
        """
        Reconstructs the 128-d L2-normalized float32 vector for post_id directly from the SQ8 index.
        Returns None if post_id is not in index or index does not support reconstruction.
        """
        if not self.is_ready():
            return None
        row_idx = self.find_row_by_post_id(post_id)
        if row_idx is None:
            return None
        try:
            vec = self.index.reconstruct(row_idx)
            vec = np.asarray(vec, dtype=np.float32)
            norm = float(np.linalg.norm(vec))
            if norm > 1e-6:
                vec = vec / norm
            return vec
        except Exception:
            return None

    def search_by_post_id(self, post_id: int, top_k: int = 30) -> Tuple[List[int], List[float]]:
        """
        Finds similar posts given an existing post_id.
        Reconstructs post vector from index or searches nearest neighbors.
        """
        if not self.is_ready():
            return [], []

        vec = self.reconstruct_post_vector(post_id)
        if vec is None:
            return [], []

        pids, sims = self.search_vector(vec, top_k=top_k + 1)
        # Filter out the query post itself
        filtered_pids = []
        filtered_sims = []
        for p, s in zip(pids, sims):
            if p != post_id:
                filtered_pids.append(p)
                filtered_sims.append(s)
            if len(filtered_pids) >= top_k:
                break
        return filtered_pids, filtered_sims
