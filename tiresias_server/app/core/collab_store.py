from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl


class CollabStore:
    """
    In-memory store for collaborative filtering assets:
    - 64 Taste Centroids (taste_centroids.npy) for persona mapping
    - Taste Archetype Top Posts (taste_archetypes.parquet) for instant cold-start
    - Post Co-favorited graph (post_cofav.parquet) for collaborative candidate generation
    """

    def __init__(
        self,
        centroids_path: Path,
        archetypes_path: Path,
        cofav_path: Path,
    ) -> None:
        self.centroids_path = centroids_path
        self.archetypes_path = archetypes_path
        self.cofav_path = cofav_path

        self.centroids: Optional[np.ndarray] = None  # (K, dim)
        self.archetype_posts: Dict[int, List[int]] = {}  # archetype_id -> [post_ids]
        self.cofav_edges: Dict[int, List[Tuple[int, float]]] = {}  # post_id -> [(neighbor_pid, weight)]

        self._load()

    def _load(self) -> None:
        # 1. Load taste centroids
        if self.centroids_path.exists():
            try:
                self.centroids = np.load(self.centroids_path.as_posix()).astype(np.float32, copy=False)
            except Exception:
                self.centroids = None

        # 2. Load taste archetypes
        if self.archetypes_path.exists():
            try:
                arch_df = pl.read_parquet(self.archetypes_path.as_posix())
                for rec in arch_df.iter_rows(named=True):
                    aid = int(rec["archetype_id"])
                    pid = int(rec["post_id"])
                    if aid not in self.archetype_posts:
                        self.archetype_posts[aid] = []
                    self.archetype_posts[aid].append(pid)
            except Exception:
                pass

        # 3. Load cofav graph edges (sample or top edges for quick lookup)
        if self.cofav_path.exists():
            try:
                cofav_df = pl.read_parquet(self.cofav_path.as_posix())
                for rec in cofav_df.iter_rows(named=True):
                    a = int(rec["a_post_id"])
                    b = int(rec["b_post_id"])
                    w = float(rec["weight"])
                    # Store undirected adjacency
                    if a not in self.cofav_edges:
                        self.cofav_edges[a] = []
                    self.cofav_edges[a].append((b, w))

                    if b not in self.cofav_edges:
                        self.cofav_edges[b] = []
                    self.cofav_edges[b].append((a, w))
            except Exception:
                pass

    def get_archetype_centroid(self, archetype_id: int) -> Optional[np.ndarray]:
        if self.centroids is None or archetype_id < 0 or archetype_id >= len(self.centroids):
            return None
        return self.centroids[archetype_id]

    def get_archetype_posts(self, archetype_id: int, limit: int = 100) -> List[int]:
        posts = self.archetype_posts.get(archetype_id, [])
        return posts[:limit]

    def find_nearest_archetype(self, vec: np.ndarray) -> int:
        if self.centroids is None or len(self.centroids) == 0:
            return 0
        v = vec.reshape(-1).astype(np.float32, copy=False)
        norm = float(np.linalg.norm(v))
        if norm > 1e-6:
            v = v / norm
        sims = self.centroids @ v
        return int(np.argmax(sims))

    def get_cofav_candidates(self, seed_post_ids: List[int], top_k: int = 50) -> List[Tuple[int, float]]:
        """Finds co-favorited neighbor posts for a list of seed post IDs."""
        candidate_scores: Dict[int, float] = {}
        seed_set = set(seed_post_ids)

        for spid in seed_post_ids:
            edges = self.cofav_edges.get(spid, [])
            for nb_pid, w in edges:
                if nb_pid in seed_set:
                    continue
                candidate_scores[nb_pid] = candidate_scores.get(nb_pid, 0.0) + w

        sorted_cand = sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True)
        return sorted_cand[:top_k]
