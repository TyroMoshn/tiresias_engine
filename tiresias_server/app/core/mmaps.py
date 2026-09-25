from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


class MmapsManager:
    """
    Zero-copy memory-mapped access to post metadata arrays.
    All arrays share the exact same row alignment pi(post_id) -> dense_idx in [0, N-1].
    Reads are serviced directly from the OS Page Cache without inflating process RSS.
    """

    def __init__(self, mmaps_dir: Path) -> None:
        self.mmaps_dir = mmaps_dir
        self.post_ids: Optional[np.ndarray] = None
        self.score: Optional[np.ndarray] = None
        self.fav_count: Optional[np.ndarray] = None
        self.rating: Optional[np.ndarray] = None
        self.w: Optional[np.ndarray] = None
        self.h: Optional[np.ndarray] = None
        self.uploader_id: Optional[np.ndarray] = None
        self.post_in_pools: Optional[np.ndarray] = None
        self.epoch_day: Optional[np.ndarray] = None
        self.file_ext: Optional[np.ndarray] = None
        self.total_posts: int = 0

        self._load()

    def _open_mmap(self, filename: str, dtype: np.dtype) -> Optional[np.ndarray]:
        path = self.mmaps_dir / filename
        if not path.exists():
            return None
        try:
            return np.memmap(path.as_posix(), mode="r", dtype=dtype)
        except Exception:
            return None

    def _load(self) -> None:
        if not self.mmaps_dir.exists():
            return

        self.post_ids = self._open_mmap("post_ids.bin", np.int64)
        if self.post_ids is not None:
            self.total_posts = len(self.post_ids)

        self.score = self._open_mmap("score.bin", np.int32)
        self.fav_count = self._open_mmap("fav_count.bin", np.int32)
        self.rating = self._open_mmap("rating.bin", np.uint8)
        self.epoch_day = self._open_mmap("epoch_day.bin", np.int32)
        self.w = self._open_mmap("w.bin", np.int16)
        if self.w is None:
            self.w = self._open_mmap("w.bin", np.int32)
        self.h = self._open_mmap("h.bin", np.int16)
        if self.h is None:
            self.h = self._open_mmap("h.bin", np.int32)
        self.uploader_id = self._open_mmap("uploader_id.bin", np.int64)
        self.post_in_pools = self._open_mmap("post_in_pools_count.bin", np.int32)
        self.file_ext = self._open_mmap("file_ext.bin", np.uint8)
        self.is_deleted = self._open_mmap("is_deleted.bin", np.uint8)

    def find_dense_idx(self, post_id: int) -> Optional[int]:
        """Binary search O(log N) for dense index of a 64-bit post ID."""
        if self.post_ids is None or self.total_posts == 0:
            return None
        idx = int(np.searchsorted(self.post_ids, post_id))
        if idx < self.total_posts and self.post_ids[idx] == post_id:
            return idx
        return None

    def get_post_id(self, dense_idx: int) -> Optional[int]:
        """O(1) lookup of 64-bit post ID by dense index."""
        if self.post_ids is None or dense_idx < 0 or dense_idx >= self.total_posts:
            return None
        return int(self.post_ids[dense_idx])

    def get_metadata(self, dense_idx: int) -> Dict[str, Any]:
        """Extracts post metadata for scoring and response formatting."""
        if dense_idx < 0 or dense_idx >= self.total_posts:
            return {}

        pid = int(self.post_ids[dense_idx]) if self.post_ids is not None else 0
        sc = int(self.score[dense_idx]) if self.score is not None else 0
        fc = int(self.fav_count[dense_idx]) if self.fav_count is not None else 0
        r_code = int(self.rating[dense_idx]) if self.rating is not None else 0
        w_val = int(self.w[dense_idx]) if self.w is not None else 0
        h_val = int(self.h[dense_idx]) if self.h is not None else 0
        u_id = int(self.uploader_id[dense_idx]) if self.uploader_id is not None else 0
        pools_cnt = int(self.post_in_pools[dense_idx]) if self.post_in_pools is not None else 0
        ed = int(self.epoch_day[dense_idx]) if self.epoch_day is not None else 0

        # Rating code mapping: 0 -> 's', 1 -> 'q', 2 -> 'e'
        r_str = "s" if r_code == 0 else ("q" if r_code == 1 else "e")

        # Media extension mapping: 0: png, 1: jpg, 2: webp, 3: gif, 4: webm, 5: mp4, 6: swf, 7: unknown
        ext_code = int(self.file_ext[dense_idx]) if self.file_ext is not None else 0
        code_to_ext = {0: "png", 1: "jpg", 2: "webp", 3: "gif", 4: "webm", 5: "mp4", 6: "swf"}
        ext_str = code_to_ext.get(ext_code, "png")
        # Video: gif, webm, mp4
        is_vid = ext_code in (3, 4, 5)
        is_del = bool(self.is_deleted[dense_idx]) if self.is_deleted is not None else False

        return {
            "post_id": pid,
            "dense_idx": dense_idx,
            "score": sc,
            "fav_count": fc,
            "epoch_day": ed,
            "rating": r_str,
            "rating_code": r_code,
            "width": w_val,
            "height": h_val,
            "uploader_id": u_id,
            "pools_count": pools_cnt,
            "file_ext": ext_str,
            "is_video": is_vid,
            "is_deleted": is_del,
        }

    def close(self) -> None:
        """Closes memory-mapped file handles if open to prevent resource leakage."""
        attrs = [
            "post_ids",
            "score",
            "fav_count",
            "rating",
            "epoch_day",
            "w",
            "h",
            "uploader_id",
            "post_in_pools",
            "file_ext",
            "is_deleted",
        ]
        for attr in attrs:
            arr = getattr(self, attr, None)
            if arr is not None:
                # np.memmap base attribute or _mmap holds the underlying mmap object
                m = getattr(arr, "_mmap", None)
                if m is not None:
                    try:
                        m.close()
                    except Exception:
                        pass
                setattr(self, attr, None)
        self.total_posts = 0
