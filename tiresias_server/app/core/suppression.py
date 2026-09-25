from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

try:
    from pyroaring import BitMap  # type: ignore
except ImportError:
    BitMap = None  # type: ignore

logger = logging.getLogger(__name__)


class SuppressionManager:
    """
    Manages cold-start tag suppression rules with configurable weights [0.0 - 1.0].
    - Weight 1.0: Hard ban from personal recommendations (feed, similar, board recs).
    - Weight 0.0 < W < 1.0: Soft score penalty modifier during candidate ranking.
    - Smooth decay: As the user likes posts containing a suppressed tag,
      the effective penalty fades to 0.0 after reaching `likes_to_unsuppress` likes.
    - Hot reload: Rules can be updated on disk and reloaded on the fly.
    """

    def __init__(
        self,
        config_path: Path,
        tags_parquet_path: Path,
        post_tags_parquet_dir: Path,
        mmaps_manager: Any,
    ) -> None:
        self.config_path = config_path
        self.tags_parquet_path = tags_parquet_path
        self.post_tags_parquet_dir = post_tags_parquet_dir
        self.mmaps = mmaps_manager

        # Settings
        self.likes_to_unsuppress: int = 5
        self.soft_penalty_multiplier: float = 0.8

        # In-memory mappings
        self.base_weights: Dict[str, float] = {}  # tag_name -> float in [0.0, 1.0]
        self.tag_name_to_id: Dict[str, int] = {}  # tag_name -> tag_id
        self.tag_id_to_name: Dict[int, str] = {}  # tag_id -> tag_name
        self.tag_bitmaps: Dict[str, BitMap] = {}  # tag_name -> BitMap of dense post indices

        self.reload()

    def reload(self, config_path: Optional[Path] = None) -> Dict[str, Any]:
        """Loads or reloads initial_suppression.json and rebuilds tag bitmaps."""
        if config_path is not None:
            self.config_path = config_path

        if not self.config_path.exists():
            logger.info("Suppression config not found at %s; operating with empty rules.", self.config_path)
            self._reset_rules()
            return {"status": "ok", "loaded_tags_count": 0, "tags": {}}

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.error("Failed parsing %s: %s", self.config_path, e)
            return {"status": "error", "error": str(e)}

        settings = data.get("settings", {})
        self.likes_to_unsuppress = max(1, int(settings.get("likes_to_unsuppress", 5)))
        self.soft_penalty_multiplier = max(0.1, min(1.0, float(settings.get("soft_penalty_multiplier", 0.8))))

        raw_tags = data.get("tags", {})
        if not isinstance(raw_tags, dict):
            raw_tags = {}

        new_weights: Dict[str, float] = {}
        for k, v in raw_tags.items():
            k_norm = str(k).strip().lower()
            if not k_norm:
                continue
            try:
                w = float(v)
                if w > 0.0:
                    new_weights[k_norm] = min(1.0, max(0.0, w))
            except (ValueError, TypeError):
                continue

        if not new_weights:
            self._reset_rules()
            logger.info("Suppression rules reloaded: 0 active tags configured.")
            return {"status": "ok", "loaded_tags_count": 0, "tags": {}}

        # Resolve tag names to tag_ids from tags.parquet
        resolved_name_to_id, resolved_id_to_name = self._resolve_tag_ids(list(new_weights.keys()))

        # Build bitmaps for resolved tags
        new_bitmaps = self._load_tag_bitmaps(resolved_name_to_id)

        # Atomic swap
        self.base_weights = new_weights
        self.tag_name_to_id = resolved_name_to_id
        self.tag_id_to_name = resolved_id_to_name
        self.tag_bitmaps = new_bitmaps

        logger.info(
            "Suppression rules reloaded: %d tags configured, %d resolved to index bitmaps.",
            len(self.base_weights),
            len(self.tag_bitmaps),
        )
        return {
            "status": "ok",
            "loaded_tags_count": len(self.base_weights),
            "resolved_tags_count": len(self.tag_bitmaps),
            "tags": self.base_weights,
            "likes_to_unsuppress": self.likes_to_unsuppress,
        }

    def _reset_rules(self) -> None:
        self.base_weights.clear()
        self.tag_name_to_id.clear()
        self.tag_id_to_name.clear()
        self.tag_bitmaps.clear()

    def _resolve_tag_ids(self, tag_names: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
        if not self.tags_parquet_path.exists() or not tag_names:
            return {}, {}

        try:
            import polars as pl

            df = (
                pl.read_parquet(self.tags_parquet_path.as_posix())
                .filter(pl.col("tag").is_in(tag_names))
                .select(["tag", "tag_id"])
            )
            name_to_id: Dict[str, int] = {}
            id_to_name: Dict[int, str] = {}
            for row in df.iter_rows():
                t_name, t_id = str(row[0]), int(row[1])
                name_to_id[t_name] = t_id
                id_to_name[t_id] = t_name
            return name_to_id, id_to_name
        except Exception as e:
            logger.error("Failed resolving tag_ids from %s: %s", self.tags_parquet_path, e)
            return {}, {}

    def _load_tag_bitmaps(self, name_to_id: Dict[str, int]) -> Dict[str, BitMap]:
        if BitMap is None or not name_to_id or not self.post_tags_parquet_dir.exists():
            return {}

        import glob
        import polars as pl

        bitmaps: Dict[str, BitMap] = {}
        for t_name in name_to_id:
            bitmaps[t_name] = BitMap()

        # Group target tag_ids by shard: tag_shard = tag_id % 256
        shard_to_tags: Dict[int, Dict[int, str]] = {}
        for t_name, t_id in name_to_id.items():
            shard = t_id % 256
            if shard not in shard_to_tags:
                shard_to_tags[shard] = {}
            shard_to_tags[shard][t_id] = t_name

        for shard, tags_in_shard in shard_to_tags.items():
            pattern = (self.post_tags_parquet_dir / f"tag_shard={shard}" / "*.parquet").as_posix()
            files = glob.glob(pattern)
            if not files:
                continue

            try:
                df = (
                    pl.read_parquet(files[0])
                    .filter(pl.col("tag_id").is_in(list(tags_in_shard.keys())))
                    .select(["post_id", "tag_id"])
                )
                for row in df.iter_rows():
                    pid, tid = int(row[0]), int(row[1])
                    t_name = tags_in_shard.get(tid)
                    if not t_name:
                        continue
                    didx = self.mmaps.find_dense_idx(pid)
                    if didx is not None:
                        bitmaps[t_name].add(didx)
            except Exception as e:
                logger.error("Error reading post_tags shard %d: %s", shard, e)

        return bitmaps

    def get_user_effective_weights(self, user_tag_likes: Dict[int, int]) -> Dict[str, float]:
        """
        Calculates effective weight per configured tag for a specific user:
        W_eff = W_base * max(0.0, 1.0 - (likes / likes_to_unsuppress)).
        """
        effective: Dict[str, float] = {}
        for t_name, w_base in self.base_weights.items():
            t_id = self.tag_name_to_id.get(t_name)
            likes = user_tag_likes.get(t_id, 0) if t_id is not None else 0
            decay = max(0.0, 1.0 - (float(likes) / float(self.likes_to_unsuppress)))
            w_eff = w_base * decay
            if w_eff > 0.001:
                effective[t_name] = w_eff
        return effective

    def get_user_hard_ban_mask(self, user_tag_likes: Dict[int, int]) -> Optional[BitMap]:
        """
        Returns a union Roaring Bitmap of all posts containing at least one
        tag that is currently hard-banned (effective weight >= 0.999) for this user.
        """
        if BitMap is None or not self.base_weights:
            return None

        combined = BitMap()
        has_any = False
        eff_weights = self.get_user_effective_weights(user_tag_likes)

        for t_name, w_eff in eff_weights.items():
            if w_eff >= 0.999:
                bm = self.tag_bitmaps.get(t_name)
                if bm is not None and len(bm) > 0:
                    combined |= bm
                    has_any = True

        return combined if has_any else None

    def get_user_soft_penalties(self, user_tag_likes: Dict[int, int]) -> Dict[str, float]:
        """
        Returns mapping of tag_name -> effective weight for soft-suppressed tags (0.0 < W_eff < 0.999).
        """
        eff_weights = self.get_user_effective_weights(user_tag_likes)
        return {t: w for t, w in eff_weights.items() if 0.001 < w < 0.999}

    def compute_post_penalty(
        self,
        dense_idx: int,
        soft_penalties: Dict[str, float],
    ) -> Tuple[float, List[str]]:
        """
        Calculates score penalty multiplier for a candidate post based on soft-suppressed tags.
        Multiplier in range (0.0, 1.0]. 1.0 means no penalty.
        """
        if not soft_penalties:
            return 1.0, []

        matched_tags: List[str] = []
        max_penalty = 0.0

        for t_name, w_eff in soft_penalties.items():
            bm = self.tag_bitmaps.get(t_name)
            if bm is not None and dense_idx in bm:
                matched_tags.append(t_name)
                if w_eff > max_penalty:
                    max_penalty = w_eff

        if not matched_tags:
            return 1.0, []

        # Effective modifier: score * (1.0 - max_penalty * soft_penalty_multiplier)
        factor = max(0.05, 1.0 - (max_penalty * self.soft_penalty_multiplier))
        reasons = [f"suppressed_tag:{t}" for t in matched_tags]
        return factor, reasons

    def get_suppressed_tag_ids_for_post(self, dense_idx: int) -> List[int]:
        """
        Returns list of tag_ids present on this post among the configured suppression tags.
        Used to record user tag likes when a post is favorited/liked.
        """
        if not self.tag_bitmaps:
            return []

        matched_ids: List[int] = []
        for t_name, bm in self.tag_bitmaps.items():
            if dense_idx in bm:
                t_id = self.tag_name_to_id.get(t_name)
                if t_id is not None:
                    matched_ids.append(t_id)
        return matched_ids

    def get_status(self) -> Dict[str, Any]:
        """Returns summary status of suppression configuration and loaded rules."""
        return {
            "config_path": str(self.config_path),
            "configured_tags_count": len(self.base_weights),
            "resolved_tags_count": len(self.tag_bitmaps),
            "likes_to_unsuppress": self.likes_to_unsuppress,
            "soft_penalty_multiplier": self.soft_penalty_multiplier,
            "tags": dict(self.base_weights),
        }
