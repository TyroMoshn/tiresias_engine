from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from fastapi import HTTPException

from ..config import ServerConfig
from ..db.database import Database
from ..db.telemetry_db import TelemetryDatabase
from .bitmaps import BitmapsManager
from .board_engine import BoardEngine
from .collab_store import CollabStore
from .diagnostics import DiagnosticsAuditor
from .faiss_wrapper import FaissWrapper
from .mmaps import MmapsManager
from .scorer import CandidateScorer
from .suppression import SuppressionManager


class Engine:
    """
    Core Recommendation Engine singleton.
    Coordinates Retrieval -> Hard Filtering -> Scoring -> Diversity -> Boards.
    """

    def __init__(self, config: Optional[ServerConfig] = None) -> None:
        self.config = config or ServerConfig()
        self.db = Database(self.config.db_path)
        self.mmaps = MmapsManager(self.config.mmaps_dir)
        self.faiss = FaissWrapper(
            self.config.post2vec_sq8_index,
            self.config.post2vec_faiss_ids,
            num_threads=self.config.faiss_threads,
        )
        self.bitmaps = BitmapsManager(self.config.bitmaps_dir)
        self.collab = CollabStore(
            self.config.taste_centroids_npy,
            self.config.taste_archetypes_parquet,
            self.config.post_cofav_parquet,
        )
        self.scorer = CandidateScorer(
            w_vector_sim=self.config.w_vector_sim,
            w_quality_log=self.config.w_quality_log,
            w_collab=self.config.w_collab,
            w_tag_match=self.config.w_tag_match,
            seen_decay=self.config.seen_decay_factor,
        )
        self.boards = BoardEngine(
            self.db,
            self.faiss,
            self.mmaps,
            self.bitmaps,
            self.config,
        )
        self.suppression = SuppressionManager(
            config_path=self.config.initial_suppression_json,
            tags_parquet_path=self.config.tags_parquet,
            post_tags_parquet_dir=self.config.post_tags_parquet_dir,
            mmaps_manager=self.mmaps,
        )
        self.popular_by_rating: Dict[str, np.ndarray] = self._precompute_popular_by_rating()
        self.start_time = time.time()
        self._maintenance_override: bool = False
        self.telemetry_db = TelemetryDatabase(self.config.telemetry_db_path)
        self.diagnostics = DiagnosticsAuditor(self)
        self.diagnostics_report: Dict[str, Any] = self.diagnostics.run_full_audit()
        self.sanity_warnings: List[str] = self.diagnostics_report.get("warnings", [])
        self._reload_lock = threading.Lock()

    @property
    def cfg(self) -> ServerConfig:
        return self.config

    def run_diagnostics(self) -> Dict[str, Any]:
        """Runs on-demand sanity checks and audit report."""
        self.diagnostics_report = self.diagnostics.run_full_audit()
        self.sanity_warnings = self.diagnostics_report.get("warnings", [])
        return self.diagnostics_report

    def _precompute_popular_by_rating(
        self,
        top_k: int = 1000,
        mmaps: Optional[MmapsManager] = None,
        bitmaps: Optional[BitmapsManager] = None,
    ) -> Dict[str, np.ndarray]:
        popular: Dict[str, np.ndarray] = {}
        m = mmaps if mmaps is not None else self.mmaps
        b = bitmaps if bitmaps is not None else self.bitmaps
        if m.total_posts == 0 or m.fav_count is None:
            return popular
        for r, bm in b.rating_masks.items():
            if bm is None or len(bm) == 0:
                continue
            if b.not_deleted_mask is not None:
                bm = bm & b.not_deleted_mask
            idx_arr = np.fromiter(bm, dtype=np.int32)
            k = min(top_k, len(idx_arr))
            if k == 0:
                continue
            favs = m.fav_count[idx_arr]
            top_part = np.argpartition(-favs, k - 1)[:k]
            top_part = top_part[np.argsort(-favs[top_part])]
            popular[r] = idx_arr[top_part]
        return popular

    def reload_artifacts(self) -> Dict[str, Any]:
        """
        Hot-reloads mmaps, bitmaps, FAISS vector index, collab store, suppression rules,
        and popular items without process termination or Uvicorn restarts.
        Safely switches references in a thread-safe manner and supports symlink dereferencing.
        """
        with self._reload_lock:
            # 1. Resolve self.config.data_root (supports symlink dereferencing if pointing to data_current / current)
            env_root = os.environ.get("TIRESIAS_DATA_ROOT")
            candidate = Path(env_root) if env_root else self.config.data_root

            if candidate.is_symlink():
                resolved_root = candidate.resolve()
            elif (candidate / "current").exists():
                resolved_root = (candidate / "current").resolve()
            elif (candidate / "data_current").exists():
                resolved_root = (candidate / "data_current").resolve()
            elif candidate.parent.name == "versions" and (candidate.parent.parent / "current").exists():
                resolved_root = (candidate.parent.parent / "current").resolve()
            else:
                resolved_root = candidate.resolve()

            self.config.data_root = resolved_root

            # 2. Re-instantiate / reload components
            new_mmaps = MmapsManager(self.config.mmaps_dir)
            new_bitmaps = BitmapsManager(self.config.bitmaps_dir)
            new_faiss = FaissWrapper(
                self.config.faiss_index_path,
                self.config.faiss_ids_path,
                num_threads=self.config.faiss_threads,
            )
            new_collab = CollabStore(
                self.config.taste_centroids_npy,
                self.config.archetypes_parquet,
                self.config.post_cofav_parquet,
            )
            new_suppression = SuppressionManager(
                config_path=self.config.initial_suppression_json,
                tags_parquet_path=self.config.tags_parquet,
                post_tags_parquet_dir=self.config.post_tags_parquet_dir,
                mmaps_manager=new_mmaps,
            )
            new_popular = self._precompute_popular_by_rating(mmaps=new_mmaps, bitmaps=new_bitmaps)

            # 3. Safely switch references
            old_mmaps = self.mmaps
            self.mmaps = new_mmaps
            self.bitmaps = new_bitmaps
            self.faiss = new_faiss
            self.collab = new_collab
            self.suppression = new_suppression
            self.popular_by_rating = new_popular

            if hasattr(old_mmaps, "close"):
                try:
                    old_mmaps.close()
                except Exception:
                    pass

            # 4. Updates self.boards with new references to self.mmaps and self.faiss
            self.boards.mmaps = self.mmaps
            self.boards.faiss = self.faiss
            self.boards.bitmaps = self.bitmaps
            self.boards.config = self.config
            self.boards.tags_parquet_path = self.config.tags_parquet
            self.boards.posts_parquet_dir = self.config.data_root / "posts_parquet"

            # 5. Re-runs self.run_diagnostics() to update sanity warnings and audit report
            self.run_diagnostics()

            logger.info(
                "Engine artifacts successfully reloaded: %d posts indexed, FAISS ready=%s, %d archetypes",
                self.mmaps.total_posts,
                self.faiss.is_ready(),
                len(self.collab.archetype_posts),
            )

            return {
                "status": "ok",
                "message": "Artifacts reloaded successfully",
                "total_posts_indexed": self.mmaps.total_posts,
                "faiss_ready": self.faiss.is_ready(),
                "faiss_total_vectors": self.faiss.total_vectors,
                "collab_archetypes": len(self.collab.archetype_posts),
            }

    def close(self) -> None:
        """Closes all underlying resources (e.g. SQLite database, open mmaps)."""
        if hasattr(self, "db") and self.db is not None:
            self.db.close()
        if hasattr(self, "telemetry_db") and self.telemetry_db is not None:
            self.telemetry_db.close()
        if hasattr(self, "mmaps") and self.mmaps is not None and hasattr(self.mmaps, "close"):
            self.mmaps.close()

    def __enter__(self) -> Engine:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    @property
    def is_maintenance_mode(self) -> bool:
        flag_file = self.config.data_root / ".maintenance"
        return self._maintenance_override or flag_file.exists()

    def set_maintenance_mode(self, enabled: bool) -> None:
        self._maintenance_override = enabled
        flag_file = self.config.data_root / ".maintenance"
        if enabled:
            try:
                flag_file.touch(exist_ok=True)
            except Exception:
                pass
        else:
            if flag_file.exists():
                try:
                    flag_file.unlink()
                except Exception:
                    pass

    def get_status(self) -> Dict[str, Any]:
        """Returns health and status summary of loaded assets."""
        import psutil
        process = psutil.Process()
        mem_rss_mb = process.memory_info().rss / (1024 * 1024)

        is_maint = self.is_maintenance_mode
        report = getattr(self, "diagnostics_report", {})
        return {
            "status": "maintenance" if is_maint else "online",
            "maintenance": is_maint,
            "message": "Database update in progress. Recommendations are temporarily unavailable." if is_maint else "OK",
            "profile": self.config.profile,
            "uptime_seconds": round(time.time() - self.start_time, 1),
            "memory_rss_mb": round(mem_rss_mb, 1),
            "total_posts_indexed": self.mmaps.total_posts,
            "faiss_ready": self.faiss.is_ready(),
            "faiss_total_vectors": self.faiss.total_vectors,
            "faiss_threads": self.config.faiss_threads or (os.cpu_count() or 4),
            "candidate_budget": {
                "faiss_top_k": self.config.candidate_faiss_top_k,
                "collab_top_k": self.config.candidate_collab_top_k,
                "popular_top_k": self.config.candidate_popular_top_k,
                "max_candidates": self.config.max_candidates,
            },
            "collab_archetypes_loaded": len(self.collab.archetype_posts),
            "collab_centroids_loaded": len(self.collab.centroids) if self.collab.centroids is not None else 0,
            "database_path": str(self.config.db_path),
            "sanity_warnings": getattr(self, "sanity_warnings", []),
            "diagnostics_status": report.get("status", "healthy"),
            "telemetry_errors_count": self.telemetry_db.get_count() if hasattr(self, "telemetry_db") else 0,
        }


    def _update_dynamic_user_archetype(self, user_id: str) -> None:
        """
        Dynamically migrates user taste archetype based on vector projection of recent liked posts.
        Only updates if user archetype is not locked by tester mode.
        """
        if self.db is None or self.collab is None or not self.faiss.is_ready():
            return
        arch_info = self.db.get_user_archetype_info(user_id)
        if arch_info.get("locked_archetype") is not None:
            return

        liked_pids = self.db.get_user_liked_posts(user_id, limit=50)
        if not liked_pids:
            self.db.set_user_archetype(user_id, 27)
            return

        vectors = []
        for pid in liked_pids:
            v = self.faiss.reconstruct_post_vector(pid)
            if v is not None:
                vectors.append(v)

        if not vectors:
            return

        mean_vec = np.mean(vectors, axis=0).astype(np.float32)
        norm = float(np.linalg.norm(mean_vec))
        if norm > 1e-6:
            mean_vec = mean_vec / norm
        nearest_arch = self.collab.find_nearest_archetype(mean_vec)
        self.db.set_user_archetype(user_id, nearest_arch)

    def get_user_archetype_state(self, user_id: str) -> Dict[str, Any]:
        """
        Calculates user archetype state, taste coherence R, and influence decay alpha:
        alpha = 1 / (1 + (N * R) / tau) with tau = 3.0.
        Also reconstructs the blended query vector q = alpha * c_arch + (1 - alpha) * u_likes.
        """
        arch_info = self.db.get_user_archetype_info(user_id)
        taste_arch = arch_info.get("taste_archetype_id", 27)
        locked_arch = arch_info.get("locked_archetype")
        forced_weight = arch_info.get("forced_archetype_weight")
        effective_arch = locked_arch if locked_arch is not None else taste_arch

        liked_pids = self.db.get_user_liked_posts(user_id, limit=100)
        vectors = []
        if self.faiss.is_ready():
            for pid in liked_pids:
                v = self.faiss.reconstruct_post_vector(pid)
                if v is not None:
                    vectors.append(v)

        n_likes = len(vectors)
        coherence = 0.0
        user_vec = None
        tau = 3.0

        if n_likes > 0:
            vec_sum = np.sum(vectors, axis=0).astype(np.float32)
            sum_norm = float(np.linalg.norm(vec_sum))
            coherence = float(np.clip(sum_norm / n_likes, 0.0, 1.0))
            if sum_norm > 1e-6:
                user_vec = vec_sum / sum_norm
            auto_alpha = float(1.0 / (1.0 + (n_likes * coherence) / tau))
        else:
            auto_alpha = 1.0

        if forced_weight is not None:
            alpha = float(np.clip(forced_weight, 0.0, 1.0))
        else:
            alpha = auto_alpha

        arch_centroid = self.collab.get_archetype_centroid(effective_arch) if self.collab else None
        query_vec = None

        if arch_centroid is not None and user_vec is not None:
            blended = alpha * arch_centroid + (1.0 - alpha) * user_vec
            b_norm = float(np.linalg.norm(blended))
            if b_norm > 1e-6:
                query_vec = (blended / b_norm).astype(np.float32)
            else:
                query_vec = arch_centroid
        elif user_vec is not None:
            query_vec = user_vec
        elif arch_centroid is not None:
            query_vec = arch_centroid

        return {
            "taste_archetype_id": taste_arch,
            "locked_archetype": locked_arch,
            "forced_archetype_weight": forced_weight,
            "effective_archetype_id": effective_arch,
            "taste_coherence": round(coherence, 4),
            "archetype_influence": round(alpha, 4),
            "auto_archetype_influence": round(auto_alpha, 4),
            "liked_vectors_count": n_likes,
            "query_vec": query_vec,
        }

    def record_feedback(self, user_id: str, post_id: int, signal_type: str) -> None:
        """Records user feedback and updates user tag likes for unsuppression decay."""
        self.db.record_feedback(user_id=user_id, post_id=post_id, signal_type=signal_type)
        st = signal_type.lower().strip()
        if st == "like":
            didx = self.mmaps.find_dense_idx(int(post_id))
            if didx is not None:
                suppressed_tag_ids = self.suppression.get_suppressed_tag_ids_for_post(didx)
                if suppressed_tag_ids:
                    self.db.increment_user_tag_likes(user_id, suppressed_tag_ids)
            self._update_dynamic_user_archetype(user_id)

    def remove_feedback(self, user_id: str, post_id: int, signal_type: Optional[str] = None) -> bool:
        """Removes user feedback (e.g. un-disliking or un-liking a post)."""
        removed = self.db.remove_feedback(user_id=user_id, post_id=post_id, signal_type=signal_type)
        if removed:
            self._update_dynamic_user_archetype(user_id)
        return removed

    def recommend_feed(
        self,
        user_id: str,
        ratings: Optional[List[str]] = None,
        media_types: Optional[List[str]] = None,
        limit: int = 30,
        cursor: int = 0,
        min_score: int = 0,
        exclude_tags: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Retrieves a personalized feed of recommended posts for a user.
        """
        if self.is_maintenance_mode:
            raise HTTPException(
                status_code=503,
                detail="Database update in progress. Recommendations will be available in a few minutes.",
            )

        t0 = time.perf_counter()
        limit = min(max(1, limit), self.config.max_limit)
        active_ratings = ratings or ["s", "g"]

        # 1. Determine user profile & query vector
        self.db.ensure_user(user_id)
        liked_pids = self.db.get_user_liked_posts(user_id, limit=20)
        hard_rejected = self.db.get_user_hard_rejected_posts(user_id)
        seen_pids = self.db.get_user_seen_posts(user_id)
        user_tag_bl = self.db.get_user_tag_blacklist(user_id)
        if exclude_tags:
            user_tag_bl.update(exclude_tags)

        # Suppression rules & decay for this user
        user_tag_likes = self.db.get_user_tag_likes(user_id)
        hard_ban_mask = self.suppression.get_user_hard_ban_mask(user_tag_likes)
        soft_penalties = self.suppression.get_user_soft_penalties(user_tag_likes)

        # Candidate dict: post_id -> {vector_sim, collab_weight}
        candidates: Dict[int, Dict[str, float]] = {}

        # 2. Stage 1: Candidate Generation
        # Channel A: Collaborative neighbors from recent likes
        if liked_pids:
            collab_cands = self.collab.get_cofav_candidates(liked_pids, top_k=self.config.candidate_collab_top_k)
            for pid, cw in collab_cands:
                candidates[pid] = {"vector_sim": 0.5, "collab_weight": float(cw)}

        # Channel B: Semantic vector retrieval with adaptive archetype decay & taste blending
        arch_state = self.get_user_archetype_state(user_id)
        user_arch = arch_state["effective_archetype_id"]
        alpha = arch_state["archetype_influence"]
        query_vec = arch_state["query_vec"]

        # Scale retrieval depth so users with many seen posts still retrieve plenty of unseen candidates
        faiss_top_k = max(
            self.config.candidate_faiss_top_k,
            min(1500, self.config.candidate_faiss_top_k + len(seen_pids) // 2 + cursor * 2),
        )

        # On cursor == 0 (refresh), if user has multiple likes, introduce slight exploratory perturbation
        if cursor == 0 and liked_pids and len(liked_pids) >= 3 and query_vec is not None:
            import random
            sample_count = min(len(liked_pids), 6)
            sampled_likes = random.sample(liked_pids, sample_count)
            sampled_vecs = [self.faiss.reconstruct_post_vector(p) for p in sampled_likes]
            valid_sampled = [v for v in sampled_vecs if v is not None]
            if valid_sampled:
                s_mean = np.mean(valid_sampled, axis=0).astype(np.float32)
                s_norm = float(np.linalg.norm(s_mean))
                if s_norm > 1e-6:
                    s_vec = s_mean / s_norm
                    perturbed = 0.75 * query_vec + 0.25 * s_vec
                    p_norm = float(np.linalg.norm(perturbed))
                    if p_norm > 1e-6:
                        query_vec = (perturbed / p_norm).astype(np.float32)

        if query_vec is not None and self.faiss.is_ready():
            faiss_pids, faiss_sims = self.faiss.search_vector(query_vec, top_k=faiss_top_k)
            for pid, sim in zip(faiss_pids, faiss_sims):
                if pid not in candidates:
                    candidates[pid] = {"vector_sim": sim, "collab_weight": 0.0}
                else:
                    candidates[pid]["vector_sim"] = max(candidates[pid]["vector_sim"], sim)

        # Channel C: Archetype top posts (scaled by archetype influence alpha)
        channel_c_budget = int(round(100 * alpha))
        if channel_c_budget > 0:
            arch_posts = self.collab.get_archetype_posts(user_arch, limit=channel_c_budget)
            for pid in arch_posts:
                if pid not in candidates:
                    candidates[pid] = {"vector_sim": 0.6, "collab_weight": 0.2}

        # Channel D: Precomputed high-quality popular candidates matching requested ratings
        for r in active_ratings:
            pop_indices = self.popular_by_rating.get(r.lower(), np.array([], dtype=np.int32))
            slice_start = (cursor * 2) % max(1, len(pop_indices) - limit) if len(pop_indices) > limit else 0
            for didx in pop_indices[slice_start : slice_start + 80]:
                pid = self.mmaps.get_post_id(int(didx))
                if pid is not None and pid not in candidates:
                    candidates[pid] = {"vector_sim": 0.45, "collab_weight": 0.15}

        # Fallback if candidates pool is small: sample from mmaps
        if len(candidates) < limit * 2 and self.mmaps.total_posts > 0:
            sample_step = max(1, self.mmaps.total_posts // 200)
            for idx in range(0, min(self.mmaps.total_posts, 200 * sample_step), sample_step):
                pid = self.mmaps.get_post_id(idx)
                if pid is not None and pid not in candidates:
                    candidates[pid] = {"vector_sim": 0.3, "collab_weight": 0.0}

        # 3. Stage 2: Hard Filtering
        filter_mask = self.bitmaps.get_filter_mask(
            allowed_ratings=active_ratings,
            allowed_media_types=media_types,
        )
        allowed_rating_codes = {self.config.rating_map.get(r.lower(), 0) for r in active_ratings}

        valid_pool: List[Dict[str, Any]] = []

        for pid, cand_meta in candidates.items():
            # Hard exclusion: disliked or hidden
            if pid in hard_rejected:
                continue

            dense_idx = self.mmaps.find_dense_idx(pid)
            if dense_idx is None:
                continue

            # Hard suppression: post contains a tag hard-banned for this user
            if hard_ban_mask is not None and dense_idx in hard_ban_mask:
                continue

            meta = self.mmaps.get_metadata(dense_idx)
            # Never show deleted or flash (swf) posts
            if meta.get("is_deleted", False) or meta.get("file_ext") == "swf":
                continue

            # Rating & media hard filter
            if meta.get("rating_code", 0) not in allowed_rating_codes:
                continue
            if filter_mask is not None and dense_idx not in filter_mask:
                continue

            # Minimum score hard filter
            if meta.get("score", 0) < min_score:
                continue

            valid_pool.append({
                "id": pid,
                "post_id": pid,
                "score_val": meta.get("score", 0),
                "fav_count": meta.get("fav_count", 0),
                "rating": meta.get("rating", "s"),
                "uploader_id": meta.get("uploader_id", 0),
                "vector_sim": cand_meta["vector_sim"],
                "collab_weight": cand_meta["collab_weight"],
                "is_seen": pid in seen_pids,
                "width": meta.get("width", 0),
                "height": meta.get("height", 0),
                "file_ext": meta.get("file_ext", "png"),
                "is_video": meta.get("is_video", False),
            })

        # Replenishment: if valid_pool is smaller than required limit + cursor, top up from rating popular lists
        if len(valid_pool) < (limit + cursor + 40):
            for r in active_ratings:
                pop_indices = self.popular_by_rating.get(r.lower(), np.array([], dtype=np.int32))
                for didx in pop_indices:
                    if filter_mask is not None and int(didx) not in filter_mask:
                        continue
                    # Skip hard banned posts
                    if hard_ban_mask is not None and int(didx) in hard_ban_mask:
                        continue
                    pid = self.mmaps.get_post_id(int(didx))
                    if pid is None or pid in hard_rejected or any(x["post_id"] == pid for x in valid_pool):
                        continue
                    meta = self.mmaps.get_metadata(int(didx))
                    if meta.get("is_deleted", False) or meta.get("file_ext") == "swf":
                        continue
                    if meta.get("score", 0) < min_score:
                        continue
                    valid_pool.append({
                        "id": pid,
                        "post_id": pid,
                        "score_val": meta.get("score", 0),
                        "fav_count": meta.get("fav_count", 0),
                        "rating": meta.get("rating", "s"),
                        "uploader_id": meta.get("uploader_id", 0),
                        "vector_sim": 0.40,
                        "collab_weight": 0.10,
                        "is_seen": pid in seen_pids,
                        "width": meta.get("width", 0),
                        "height": meta.get("height", 0),
                        "file_ext": meta.get("file_ext", "png"),
                        "is_video": meta.get("is_video", False),
                    })
                    if len(valid_pool) >= (limit + cursor + 80):
                        break

        # 4. Stage 3: Multi-factor Scoring
        scored_items: List[Dict[str, Any]] = []
        for item in valid_pool:
            final_sc, reasons = self.scorer.score_candidate(
                vector_sim=item["vector_sim"],
                fav_count=item["fav_count"],
                score_val=item["score_val"],
                collab_weight=item["collab_weight"],
                is_seen=item["is_seen"],
            )
            # Soft suppression penalty modifier
            didx = self.mmaps.find_dense_idx(item["post_id"])
            if didx is not None and soft_penalties:
                penalty_factor, supp_reasons = self.suppression.compute_post_penalty(didx, soft_penalties)
                final_sc *= penalty_factor
                reasons.extend(supp_reasons)

            item["score"] = final_sc
            item["reasons"] = reasons
            scored_items.append(item)

        # 5. Stage 4: Freshness Tiering, Diversity, and Pagination
        # Prioritize unseen candidates so previously seen posts never block fresh recommendations
        unseen_items = [it for it in scored_items if not it.get("is_seen", False)]
        seen_items = [it for it in scored_items if it.get("is_seen", False)]

        needed_count = limit + cursor + 1
        diversified_unseen = self.scorer.diversify_and_rank(unseen_items, limit=needed_count)

        if len(diversified_unseen) >= needed_count:
            diversified = diversified_unseen
        else:
            remaining_slots = needed_count - len(diversified_unseen)
            diversified_seen = self.scorer.diversify_and_rank(seen_items, limit=remaining_slots)
            diversified = diversified_unseen + diversified_seen

        paged_items = diversified[cursor : cursor + limit]
        has_more = len(diversified) > (cursor + len(paged_items))

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "items": paged_items,
            "total_candidates": len(candidates),
            "filtered_pool_size": len(valid_pool),
            "returned_count": len(paged_items),
            "cursor": cursor + len(paged_items),
            "has_more": has_more,
            "latency_ms": round(latency_ms, 2),
        }

    def recommend_similar(
        self,
        post_id: int,
        ratings: Optional[List[str]] = None,
        media_types: Optional[List[str]] = None,
        limit: int = 20,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Retrieves visually and semantically similar posts for a given post_id.
        """
        if self.is_maintenance_mode:
            raise HTTPException(
                status_code=503,
                detail="Database update in progress. Recommendations will be available in a few minutes.",
            )

        t0 = time.perf_counter()
        limit = min(max(1, limit), self.config.max_limit)
        active_ratings = ratings or ["s", "g"]
        allowed_rating_codes = {self.config.rating_map.get(r.lower(), 0) for r in active_ratings}

        user_tag_likes = self.db.get_user_tag_likes(user_id) if user_id else {}
        hard_ban_mask = self.suppression.get_user_hard_ban_mask(user_tag_likes)

        sim_pids, sim_scores = self.faiss.search_by_post_id(post_id, top_k=limit * 3)
        filter_mask = self.bitmaps.get_filter_mask(
            allowed_ratings=active_ratings,
            allowed_media_types=media_types,
        )

        items: List[Dict[str, Any]] = []
        for pid, sim in zip(sim_pids, sim_scores):
            dense_idx = self.mmaps.find_dense_idx(pid)
            if dense_idx is not None:
                # Hard suppression: skip posts with tags hard-banned for this user
                if hard_ban_mask is not None and dense_idx in hard_ban_mask:
                    continue
                meta = self.mmaps.get_metadata(dense_idx)
                if meta.get("is_deleted", False) or meta.get("file_ext") == "swf":
                    continue
                if meta.get("rating_code", 0) not in allowed_rating_codes:
                    continue
                if filter_mask is not None and dense_idx not in filter_mask:
                    continue
                items.append({
                    "id": pid,
                    "post_id": pid,
                    "similarity": round(float(sim), 4),
                    "score": meta.get("score", 0),
                    "fav_count": meta.get("fav_count", 0),
                    "rating": meta.get("rating", "s"),
                    "width": meta.get("width", 0),
                    "height": meta.get("height", 0),
                    "file_ext": meta.get("file_ext", "png"),
                    "is_video": meta.get("is_video", False),
                })
            else:
                if self.mmaps.total_posts == 0:
                    items.append({
                        "id": pid,
                        "post_id": pid,
                        "similarity": round(float(sim), 4),
                        "rating": active_ratings[0],
                    })

            if len(items) >= limit:
                break

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "query_post_id": post_id,
            "items": items,
            "returned_count": len(items),
            "latency_ms": round(latency_ms, 2),
        }

    def close(self) -> None:
        """Closes all database connections and releases file locks."""
        if hasattr(self, "db") and self.db is not None:
            self.db.close()
        if hasattr(self, "telemetry_db") and self.telemetry_db is not None:
            self.telemetry_db.close()

    def __enter__(self) -> Engine:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

