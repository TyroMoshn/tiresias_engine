from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set, Tuple


class CandidateScorer:
    """
    Computes multi-factor ranking scores for retrieved candidate posts:
    Score = w_vec * VectorSim + w_qual * QualityLog + w_collab * CollabBonus
    Applies soft score decay for previously seen posts and diversity re-ranking.
    """

    def __init__(
        self,
        w_vector_sim: float = 0.50,
        w_quality_log: float = 0.25,
        w_collab: float = 0.15,
        w_tag_match: float = 0.10,
        seen_decay: float = 0.30,
    ) -> None:
        self.w_vector_sim = w_vector_sim
        self.w_quality_log = w_quality_log
        self.w_collab = w_collab
        self.w_tag_match = w_tag_match
        self.seen_decay = seen_decay
        self.max_fav_log = math.log(1.0 + 100_000.0)  # Reference normalizing bound

    def score_candidate(
        self,
        vector_sim: float,
        fav_count: int,
        score_val: int,
        collab_weight: float = 0.0,
        tag_match_score: float = 0.0,
        is_seen: bool = False,
    ) -> Tuple[float, List[str]]:
        reasons: List[str] = []

        # 1. Vector semantic similarity component
        vec_component = max(0.0, float(vector_sim))
        if vec_component > 0.70:
            reasons.append("visual_affinity")

        # 2. Logarithmic quality component
        norm_fav = math.log(1.0 + max(0, fav_count)) / self.max_fav_log
        qual_component = min(1.0, norm_fav)
        if fav_count >= 500:
            reasons.append("community_favorite")

        # 3. Collab component
        collab_component = min(1.0, float(collab_weight))
        if collab_component > 0.3:
            reasons.append("co_favorited")

        # 4. Tag match component
        tag_component = min(1.0, float(tag_match_score))
        if tag_component > 0.5:
            reasons.append("tag_preference")

        # Total combined score
        final_score = (
            self.w_vector_sim * vec_component
            + self.w_quality_log * qual_component
            + self.w_collab * collab_component
            + self.w_tag_match * tag_component
        )

        # Soft decay for previously seen posts
        if is_seen:
            final_score *= self.seen_decay
            reasons.append("previously_seen")

        return final_score, reasons

    def diversify_and_rank(
        self,
        scored_candidates: List[Dict[str, Any]],
        limit: int = 30,
        max_consecutive_uploader: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        Sorts candidates by final score and enforces diversity:
        No more than `max_consecutive_uploader` posts from the same author sequentially.
        """
        # Sort descending by score
        scored_candidates.sort(key=lambda item: item["score"], reverse=True)

        selected: List[Dict[str, Any]] = []
        deferred: List[Dict[str, Any]] = []
        consecutive_author_count = 0
        last_uploader_id = -1

        for cand in scored_candidates:
            u_id = cand.get("uploader_id", 0)

            if u_id > 0 and u_id == last_uploader_id:
                if consecutive_author_count >= max_consecutive_uploader:
                    deferred.append(cand)
                    continue
                else:
                    consecutive_author_count += 1
            else:
                last_uploader_id = u_id
                consecutive_author_count = 1

            selected.append(cand)
            if len(selected) >= limit:
                break

        # If more items needed to meet limit, fill from deferred
        if len(selected) < limit and deferred:
            needed = limit - len(selected)
            selected.extend(deferred[:needed])

        return selected
