from __future__ import annotations

import collections
import datetime
import glob
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import polars as pl

from ..config import ServerConfig
from ..db.database import Database
from .bitmaps import BitmapsManager
from .faiss_wrapper import FaissWrapper
from .mmaps import MmapsManager

logger = logging.getLogger("tiresias.board_engine")


class BoardEngine:
    """
    Pinterest-like dynamic Board Recommendation & Analytics Engine.
    Features:
    - Zero-copy FAISS SQ8 vector reconstruction for real-time Centroid computation.
    - Post affinity to board centroid (semantic alignment score).
    - Multi-factor server-side sorting: added_at, epoch_day, score, fav_count, affinity.
    - Salient tags extraction via TF-IDF over board post metadata.
    - 'More Like This Board' recommendations with hard rejection of existing board posts.
    """

    def __init__(
        self,
        db: Database,
        faiss: FaissWrapper,
        mmaps: MmapsManager,
        bitmaps: BitmapsManager,
        config: ServerConfig,
    ) -> None:
        self.db = db
        self.faiss = faiss
        self.mmaps = mmaps
        self.bitmaps = bitmaps
        self.config = config
        self.tags_parquet_path = config.tags_parquet
        self.posts_parquet_dir = config.data_root / "posts_parquet"

    # -------------------------------------------------------------------------
    # 1. Centroid & Affinity Calculation
    # -------------------------------------------------------------------------

    def compute_board_centroid(
        self, post_ids: Iterable[int]
    ) -> Tuple[Optional[np.ndarray], List[int], List[int]]:
        """
        Computes the L2-normalized semantic centroid vector across all valid posts in the board.
        Missing posts (e.g. newer than dataset cut-off) are logged with a warning and skipped.

        Returns:
            (centroid_vector, valid_post_ids, missing_post_ids)
        """
        valid_ids: List[int] = []
        missing_ids: List[int] = []
        vectors: List[np.ndarray] = []

        for pid in post_ids:
            v = self.faiss.reconstruct_post_vector(int(pid))
            if v is not None:
                vectors.append(v)
                valid_ids.append(int(pid))
            else:
                missing_ids.append(int(pid))
                logger.warning(
                    "Post ID %s not found in FAISS index/dataset; skipped for board centroid calculation",
                    pid,
                )

        if not vectors:
            return None, [], missing_ids

        stacked = np.stack(vectors, axis=0)
        centroid = np.mean(stacked, axis=0).astype(np.float32)
        norm = float(np.linalg.norm(centroid))
        if norm > 1e-6:
            centroid = centroid / norm

        return centroid, valid_ids, missing_ids

    def compute_affinities(
        self, post_ids: Iterable[int], centroid: Optional[np.ndarray]
    ) -> Dict[int, float]:
        """
        Computes the cosine similarity between each post in post_ids and the board centroid.
        Values range from -1.0 to 1.0. High values represent core/defining posts of the board.
        """
        if centroid is None:
            return {int(pid): 0.0 for pid in post_ids}

        affinities: Dict[int, float] = {}
        for pid in post_ids:
            vec = self.faiss.reconstruct_post_vector(int(pid))
            if vec is not None:
                dot = float(np.dot(vec, centroid))
                affinities[int(pid)] = max(-1.0, min(1.0, dot))
            else:
                affinities[int(pid)] = 0.0

        return affinities

    # -------------------------------------------------------------------------
    # 2. Salient Tags Extraction (TF-IDF)
    # -------------------------------------------------------------------------

    def extract_salient_tags(
        self, post_ids: List[int], top_k: int = 8
    ) -> List[Dict[str, Any]]:
        """
        Extracts salient keywords describing the theme of the board using TF-IDF.
        Safely falls back to empty list if posts parquet is unavailable.
        """
        if not post_ids or not self.posts_parquet_dir.exists():
            return []

        # 1. Group posts by rating, year, month to open targeted parquet partitions
        groups: Dict[Tuple[str, int, int], List[int]] = collections.defaultdict(list)
        for pid in post_ids:
            didx = self.mmaps.find_dense_idx(int(pid))
            if didx is None:
                continue
            meta = self.mmaps.get_metadata(didx)
            r = meta.get("rating", "s")
            ed = meta.get("epoch_day", 0)
            if ed > 0:
                dt = datetime.datetime.fromtimestamp(ed * 86400, datetime.timezone.utc)
                groups[(r, dt.year, dt.month)].append(int(pid))

        if not groups:
            return []

        # 2. Read tags for each post from targeted parquet partitions
        tag_post_count: Dict[str, int] = collections.defaultdict(int)
        total_found = 0

        for (r, year, month), pids in groups.items():
            pattern = (
                self.posts_parquet_dir
                / f"rating={r}"
                / f"year={year}"
                / f"month={month:02d}"
                / "*.parquet"
            )
            files = glob.glob(pattern.as_posix())
            if not files:
                continue
            try:
                df = (
                    pl.read_parquet(files[0])
                    .filter(pl.col("id").is_in(pids))
                    .select(["id", "tag_string"])
                )
                for row in df.iter_rows():
                    total_found += 1
                    tag_str = row[1]
                    if tag_str:
                        tags_set = set(tag_str.split())
                        for t in tags_set:
                            tag_post_count[t] += 1
            except Exception as e:
                logger.debug("Failed reading partition for salient tags: %s", e)

        if not tag_post_count or total_found == 0:
            return []

        # 3. Retrieve IDF for candidate tags from tags.parquet
        candidate_tags = list(tag_post_count.keys())
        tag_idfs: Dict[str, Tuple[int, float]] = {}
        if self.tags_parquet_path.exists():
            try:
                idf_df = (
                    pl.read_parquet(self.tags_parquet_path.as_posix())
                    .filter(pl.col("tag").is_in(candidate_tags))
                    .select(["tag", "category", "idf"])
                )
                for r in idf_df.iter_rows():
                    tag_idfs[r[0]] = (int(r[1]), float(r[2]) if r[2] is not None else 1.0)
            except Exception as e:
                logger.debug("Failed reading tags.parquet: %s", e)

        # 4. Score tags: TF * IDF
        scored: List[Dict[str, Any]] = []
        for t, cnt in tag_post_count.items():
            tf = cnt / total_found
            cat, idf = tag_idfs.get(t, (0, 1.0))
            score = tf * idf
            scored.append({
                "tag": t,
                "score": round(float(score), 3),
                "tf": round(float(tf), 3),
                "idf": round(float(idf), 3),
                "category": cat,
                "post_count": cnt,
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    # -------------------------------------------------------------------------
    # 3. Hydrated Board Posts with Server-Side Sorting
    # -------------------------------------------------------------------------

    def get_hydrated_board_posts(
        self,
        board_id: str,
        sort_by: str = "added_at",
        order: str = "desc",
        limit: int = 50,
        offset: int = 0,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Retrieves, hydrates, and sorts posts for a given board.
        Supported sort_by:
          - 'added_at': Date/time post was added to the board.
          - 'epoch_day' / 'created_at': Original post publication date on site.
          - 'score': Net score on site.
          - 'fav_count': Favorite count on site.
          - 'affinity': Cosine similarity to the board's centroid (in-memory FAISS SQ8).
        """
        records = self.db.get_board_post_records(board_id)
        total_count = len(records)
        if total_count == 0:
            return [], 0

        pids = [int(r["post_id"]) for r in records]

        # Compute affinities if requested or for full metadata
        affinities: Dict[int, float] = {}
        if sort_by == "affinity":
            centroid, _, _ = self.compute_board_centroid(pids)
            affinities = self.compute_affinities(pids, centroid)

        hydrated: List[Dict[str, Any]] = []
        for r in records:
            pid = int(r["post_id"])
            added_at = float(r["added_at"])
            didx = self.mmaps.find_dense_idx(pid)
            if didx is not None:
                meta = self.mmaps.get_metadata(didx)
                score = meta.get("score", 0)
                fav_count = meta.get("fav_count", 0)
                epoch_day = meta.get("epoch_day", 0)
                rating = meta.get("rating", "s")
                width = meta.get("width", 0)
                height = meta.get("height", 0)
            else:
                score = 0
                fav_count = 0
                epoch_day = 0
                rating = "s"
                width = 0
                height = 0

            aff = affinities.get(pid, 0.0) if affinities else 0.0

            hydrated.append({
                "post_id": pid,
                "added_at": added_at,
                "score": score,
                "fav_count": fav_count,
                "epoch_day": epoch_day,
                "rating": rating,
                "width": width,
                "height": height,
                "affinity": round(aff, 4),
            })

        # Apply sorting
        reverse = (order.lower() == "desc")
        if sort_by == "added_at":
            hydrated.sort(key=lambda x: x["added_at"], reverse=reverse)
        elif sort_by in ("epoch_day", "created_at"):
            hydrated.sort(key=lambda x: x["epoch_day"], reverse=reverse)
        elif sort_by == "score":
            hydrated.sort(key=lambda x: x["score"], reverse=reverse)
        elif sort_by == "fav_count":
            hydrated.sort(key=lambda x: x["fav_count"], reverse=reverse)
        elif sort_by == "affinity":
            hydrated.sort(key=lambda x: x["affinity"], reverse=reverse)
        else:
            hydrated.sort(key=lambda x: x["added_at"], reverse=reverse)

        sliced = hydrated[offset : offset + limit]
        return sliced, total_count

    # -------------------------------------------------------------------------
    # 4. Coverage Analytics & Warnings
    # -------------------------------------------------------------------------

    def get_board_coverage(self, board_id: str) -> Dict[str, Any]:
        """
        Calculates how many board posts are present in the offline vector index.
        Provides a user-facing warning if coverage is low or unindexed.
        """
        board_pids_set = self.db.get_all_board_post_ids(board_id)
        total_pids = len(board_pids_set)
        if total_pids == 0:
            return {
                "total_posts": 0,
                "indexed_posts": 0,
                "coverage_ratio": 0.0,
                "status": "empty",
                "warning_message": "Board is empty; add posts to generate recommendations.",
            }

        valid_count = 0
        for pid in board_pids_set:
            if self.faiss.reconstruct_post_vector(int(pid)) is not None:
                valid_count += 1

        ratio = round(valid_count / total_pids, 2)
        if valid_count == 0:
            status = "unindexed"
            msg = "All posts on this board are newer than the offline model cut-off. Add earlier posts (before autumn 2024) to enable semantic recommendations."
        elif valid_count < 3 and total_pids >= 3:
            status = "low_coverage"
            msg = f"Only {valid_count} of {total_pids} posts are indexed in the offline model. Recommendations may be less accurate."
        else:
            status = "sufficient"
            msg = None

        return {
            "total_posts": total_pids,
            "indexed_posts": valid_count,
            "coverage_ratio": ratio,
            "status": status,
            "warning_message": msg,
        }

    # -------------------------------------------------------------------------
    # 4. 'More Like This Board' Recommendations
    # -------------------------------------------------------------------------

    def recommend_for_board(
        self,
        board_id: str,
        user_id: str = "default_user",
        ratings: Optional[List[str]] = None,
        limit: int = 30,
        min_score: int = 0,
        exclude_tags: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Generates personalized recommendations tailored to the board's collective aesthetic.
        Crucial requirement: HARD EXCLUSION of all posts already present on this board.
        """
        t0 = time.perf_counter()
        board = self.db.get_board(board_id)
        if not board:
            return {
                "board_id": board_id,
                "error": "Board not found",
                "items": [],
                "total_candidates": 0,
                "filtered_pool_size": 0,
                "returned_count": 0,
                "latency_ms": 0.0,
            }

        board_pids_set = self.db.get_all_board_post_ids(board_id)
        total_pids = len(board_pids_set)
        if not board_pids_set:
            coverage = {
                "total_posts": 0,
                "indexed_posts": 0,
                "coverage_ratio": 0.0,
                "status": "empty",
                "warning_message": "Board is empty; add posts to generate recommendations.",
            }
            return {
                "board_id": board_id,
                "items": [],
                "total_candidates": 0,
                "filtered_pool_size": 0,
                "returned_count": 0,
                "coverage": coverage,
                "message": "Board is empty; add posts to generate recommendations.",
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            }

        # 1. Compute board centroid
        centroid, valid_pids, missing_pids = self.compute_board_centroid(board_pids_set)
        indexed_count = len(valid_pids)
        cov_ratio = round(indexed_count / max(1, total_pids), 2)
        if indexed_count == 0:
            cov_status = "unindexed"
            cov_msg = "All posts on this board are newer than the offline model cut-off. Add earlier posts (before autumn 2024) to enable semantic recommendations."
        elif indexed_count < 3 and total_pids >= 3:
            cov_status = "low_coverage"
            cov_msg = f"Only {indexed_count} of {total_pids} posts are indexed in the offline model. Recommendations may be less accurate."
        else:
            cov_status = "sufficient"
            cov_msg = None

        coverage_info = {
            "total_posts": total_pids,
            "indexed_posts": indexed_count,
            "coverage_ratio": cov_ratio,
            "status": cov_status,
            "warning_message": cov_msg,
        }

        if centroid is None:
            return {
                "board_id": board_id,
                "items": [],
                "total_candidates": 0,
                "filtered_pool_size": 0,
                "returned_count": 0,
                "coverage": coverage_info,
                "message": cov_msg or "None of the board posts exist in the semantic vector index.",
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            }

        # 2. FAISS vector search
        candidate_k = max(limit * 6, self.config.candidate_faiss_top_k)
        cand_pids, sims = self.faiss.search_vector(centroid, top_k=candidate_k)

        # 3. Hard filtering
        user_rejected = self.db.get_user_hard_rejected_posts(user_id)
        user_seen = self.db.get_user_seen_posts(user_id)

        # 3. Smart rating detection and board rating distribution
        rating_counts: Dict[str, int] = collections.defaultdict(int)
        for bpid in board_pids_set:
            bdidx = self.mmaps.find_dense_idx(int(bpid))
            if bdidx is not None:
                bmeta = self.mmaps.get_metadata(bdidx)
                br = (bmeta.get("rating") or "s").lower()
                rating_counts[br] += 1

        total_board_rated = max(1, sum(rating_counts.values()))
        p_s = rating_counts.get("s", 0) / total_board_rated
        p_q = rating_counts.get("q", 0) / total_board_rated
        p_e = rating_counts.get("e", 0) / total_board_rated

        if not ratings:
            if p_e == 0 and p_q == 0:
                # 100% S -> strictly safe
                active_ratings = ["s"]
            elif p_e == 0:
                # S + Q (no E) -> allow s and q, strictly no explicit
                active_ratings = ["s", "q"]
            else:
                # Board contains explicit content -> allow all ratings
                active_ratings = ["s", "q", "e"]
        else:
            active_ratings = ratings

        allowed_mask = self.bitmaps.get_rating_mask(active_ratings)

        filtered_pids: List[int] = []
        filtered_sims: List[float] = []

        for pid, sim in zip(cand_pids, sims):
            # HARD FILTER: Discard any post that is ALREADY on the board
            if pid in board_pids_set:
                continue
            # Discard user explicit dislikes or hides
            if pid in user_rejected:
                continue
            didx = self.mmaps.find_dense_idx(pid)
            if didx is None:
                continue
            # Rating mask check
            if allowed_mask is not None and didx not in allowed_mask:
                continue
            # Min score check
            if self.mmaps.score is not None and self.mmaps.score[didx] < min_score:
                continue

            filtered_pids.append(pid)
            filtered_sims.append(sim)

        # 4. Multi-factor scoring + Soft decay for seen
        # Semantic similarity is the DOMINANT signal (>95%).
        # Score is used strictly as a subtle normalized tie-breaker (+0.03 max) to avoid 'score waves'.
        items: List[Dict[str, Any]] = []
        max_p = max(p_s, p_q, p_e, 0.01)

        for pid, sim in zip(filtered_pids, filtered_sims):
            didx = self.mmaps.find_dense_idx(pid)
            meta = self.mmaps.get_metadata(didx) if didx is not None else {}
            sc_val = meta.get("score", 0)
            fc_val = meta.get("fav_count", 0)
            r = (meta.get("rating") or "s").lower()

            # Base score: semantic similarity to board centroid
            final_score = float(sim)

            # Rating distribution weighting (when auto-detected and board has explicit)
            if not ratings and p_e > 0:
                rating_prop = rating_counts.get(r, 0) / total_board_rated
                rating_mult = 0.88 + 0.12 * (rating_prop / max_p)
                final_score *= rating_mult

            # Subtle normalized quality tie-breaker:
            # log(1+score)/log(1+1000) in [0, 1], weighted with 0.03
            if sc_val > 0:
                norm_quality = min(1.0, float(np.log1p(float(sc_val)) / np.log1p(1000.0)))
                final_score += 0.03 * norm_quality

            # Soft decay for previously seen posts
            if pid in user_seen:
                final_score *= self.config.seen_decay_factor

            items.append({
                "post_id": pid,
                "score": round(float(final_score), 4),
                "similarity": round(float(sim), 4),
                "score_val": sc_val,
                "fav_count": fc_val,
                "rating": meta.get("rating", "s"),
                "width": meta.get("width", 0),
                "height": meta.get("height", 0),
            })

        items.sort(key=lambda x: x["score"], reverse=True)
        final_items = items[:limit]
        latency_ms = (time.perf_counter() - t0) * 1000

        return {
            "board_id": board_id,
            "items": final_items,
            "total_candidates": len(cand_pids),
            "filtered_pool_size": len(filtered_pids),
            "returned_count": len(final_items),
            "coverage": coverage_info,
            "latency_ms": round(latency_ms, 2),
        }
