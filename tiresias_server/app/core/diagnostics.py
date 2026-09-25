from __future__ import annotations

import logging
import time
from typing import Any, Dict, List
import numpy as np

logger = logging.getLogger("tiresias.diagnostics")


class DiagnosticsAuditor:
    """
    Performs comprehensive, modular sanity checks and system health audits.
    Detects data corruption, class imbalance, missing roaring bitmasks, and FAISS index anomalies.
    Does not halt execution on warnings; provides actionable diagnostic logs and API responses.
    """

    def __init__(self, engine: Any) -> None:
        self.engine = engine

    def audit_ratings(self) -> List[str]:
        warnings: List[str] = []
        mmaps = self.engine.mmaps
        bitmaps = self.engine.bitmaps

        total_posts = mmaps.total_posts
        if total_posts == 0:
            warnings.append("Total indexed posts is 0; index appears completely empty.")
            return warnings

        # 1. Check rating.bin mmap
        if mmaps.rating is None or len(mmaps.rating) == 0:
            warnings.append("rating.bin is missing or unmapped; all posts fallback to 's'.")
        elif len(mmaps.rating) != total_posts:
            warnings.append(
                f"rating.bin size mismatch: {len(mmaps.rating)} bytes vs {total_posts} total posts."
            )
        else:
            # Sample first 500k posts for rating distribution check
            sample_size = min(total_posts, 500000)
            unique, counts = np.unique(mmaps.rating[:sample_size], return_counts=True)
            dist = dict(zip(unique.tolist(), counts.tolist()))
            # 0: s, 1: q, 2: e
            if 2 not in dist or dist[2] == 0:
                warnings.append("Zero explicit ('e') posts found in sampled rating.bin!")
            if 0 not in dist or dist[0] == 0:
                warnings.append("Zero safe ('s') posts found in sampled rating.bin!")

        # 2. Check roaring bitmaps
        s_bm = bitmaps.rating_masks.get("s")
        q_bm = bitmaps.rating_masks.get("q")
        e_bm = bitmaps.rating_masks.get("e")

        s_count = len(s_bm) if s_bm is not None else 0
        q_count = len(q_bm) if q_bm is not None else 0
        e_count = len(e_bm) if e_bm is not None else 0

        if s_count == 0:
            warnings.append("Rating mask 's' is empty in BitmapsManager.")
        if e_count == 0:
            warnings.append("Rating mask 'e' is empty in BitmapsManager.")
        if q_count == 0:
            warnings.append("Rating mask 'q' is empty in BitmapsManager.")

        total_masked = s_count + q_count + e_count
        if total_posts > 1000 and total_masked < total_posts * 0.90:
            warnings.append(
                f"Sum of rating bitmaps ({total_masked}) covers less than 90% of total posts ({total_posts})."
            )

        return warnings

    def audit_media_assets(self) -> List[str]:
        warnings: List[str] = []
        mmaps = self.engine.mmaps
        bitmaps = self.engine.bitmaps

        total_posts = mmaps.total_posts
        if mmaps.file_ext is None or len(mmaps.file_ext) == 0:
            warnings.append("file_ext.bin is missing or empty; media type filtering will default to 'png'.")
        elif len(mmaps.file_ext) != total_posts:
            warnings.append(
                f"file_ext.bin size mismatch: {len(mmaps.file_ext)} vs {total_posts} total posts."
            )

        img_bm = bitmaps.media_masks.get("image")
        vid_bm = bitmaps.media_masks.get("video")
        not_del_bm = bitmaps.not_deleted_mask

        if img_bm is None or len(img_bm) == 0:
            warnings.append("media_image.roar bitmap is missing or empty.")
        if vid_bm is None or len(vid_bm) == 0:
            warnings.append("media_video.roar bitmap is missing or empty.")
        if not_del_bm is None or len(not_del_bm) == 0:
            warnings.append("not_deleted.roar bitmap is missing or empty.")

        # Invariant check: media masks must not overlap with deleted posts
        if not_del_bm is not None and len(not_del_bm) > 0:
            if img_bm is not None and not img_bm.issubset(not_del_bm):
                diff = len(img_bm - not_del_bm)
                warnings.append(f"media_image bitmap contains {diff} deleted/swf posts!")
            if vid_bm is not None and not vid_bm.issubset(not_del_bm):
                diff = len(vid_bm - not_del_bm)
                warnings.append(f"media_video bitmap contains {diff} deleted/swf posts!")

        return warnings

    def audit_faiss(self) -> List[str]:
        warnings: List[str] = []
        faiss_wrap = self.engine.faiss
        if not faiss_wrap.is_ready():
            warnings.append("FAISS vector index is not ready; vector similarity search will be unavailable.")
            return warnings

        vec_count = faiss_wrap.total_vectors
        if vec_count == 0:
            warnings.append("FAISS index loaded but contains 0 vectors.")
            return warnings

        # Test dummy vector query
        try:
            test_vec = np.zeros((1, 128), dtype=np.float32)
            test_vec[0, 0] = 1.0
            pids, similarities = faiss_wrap.search_vector(test_vec, top_k=5)
            if len(pids) == 0:
                warnings.append("FAISS search returned 0 results for test unit vector.")
            elif np.isnan(similarities).any() or np.isinf(similarities).any():
                warnings.append("FAISS search returned NaN or Inf similarity scores.")
        except Exception as ex:
            warnings.append(f"FAISS test search threw an exception: {ex}")

        return warnings

    def audit_collab(self) -> List[str]:
        warnings: List[str] = []
        collab = self.engine.collab
        num_archetypes = len(collab.archetype_posts)
        if num_archetypes == 0:
            warnings.append("Zero collaborative archetypes loaded; cold-start falls back to popularity.")
        elif num_archetypes < 64:
            warnings.append(f"Only {num_archetypes}/64 archetypes loaded.")

        if collab.centroids is None or len(collab.centroids) == 0:
            warnings.append("Archetype centroid matrix is missing.")
        elif collab.centroids.shape[1] != 128:
            warnings.append(f"Archetype centroid dimension is {collab.centroids.shape[1]} (expected 128).")

        return warnings

    def audit_suppression(self) -> List[str]:
        warnings: List[str] = []
        supp = self.engine.suppression
        if not supp.config_path.exists():
            warnings.append(f"Suppression config not found at {supp.config_path}.")
        else:
            # Check if any configured tag failed resolution
            for tag in supp.base_weights.keys():
                if tag not in supp.tag_name_to_id:
                    warnings.append(f"Tag '{tag}' configured in suppression config not found in tags database.")

        return warnings

    def audit_databases(self) -> List[str]:
        warnings: List[str] = []
        db = getattr(self.engine, "db", None)
        if db is not None:
            res = db.check_integrity()
            if res.lower() != "ok":
                warnings.append(f"tiresias_user.db PRAGMA integrity_check reported: {res}")

        t_db = getattr(self.engine, "telemetry_db", None)
        if t_db is not None:
            res = t_db.check_integrity()
            if res.lower() != "ok":
                warnings.append(f"tiresias_telemetry.db PRAGMA integrity_check reported: {res}")

        return warnings

    def run_full_audit(self) -> Dict[str, Any]:
        t0 = time.perf_counter()
        results = {
            "ratings": self.audit_ratings(),
            "media_assets": self.audit_media_assets(),
            "faiss": self.audit_faiss(),
            "collab": self.audit_collab(),
            "suppression": self.audit_suppression(),
            "databases": self.audit_databases(),
        }
        all_warnings: List[str] = []
        for cat, msgs in results.items():
            all_warnings.extend(msgs)

        duration_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        status = "healthy" if not all_warnings else "warning"

        for w in all_warnings:
            logger.warning("[SANITY AUDIT] %s", w)

        return {
            "status": status,
            "audit_timestamp": time.time(),
            "audit_duration_ms": duration_ms,
            "total_warnings": len(all_warnings),
            "warnings": all_warnings,
            "subsystems": {
                cat: {"status": "ok" if not msgs else "warning", "warnings": msgs}
                for cat, msgs in results.items()
            },
        }
