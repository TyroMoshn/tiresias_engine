from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

try:
    from pyroaring import BitMap  # type: ignore
except ImportError:
    BitMap = None  # type: ignore


class BitmapsManager:
    """
    Manages Roaring Bitmaps for high-performance bitwise hard filtering:
    - Censorship rating masks (rating_s, rating_q, rating_e)
    - Set difference for disliked/hidden/seen posts
    """

    def __init__(self, bitmaps_dir: Path) -> None:
        self.bitmaps_dir = bitmaps_dir
        self.rating_masks: Dict[str, BitMap] = {}
        self.media_masks: Dict[str, BitMap] = {}
        self.not_deleted_mask: Optional[BitMap] = None
        self._load()

    def _load(self) -> None:
        if BitMap is None:
            return

        candidate_dirs = [
            self.bitmaps_dir / "ratings",
            self.bitmaps_dir,
            self.bitmaps_dir.parent / "mmaps",
        ]

        for r_name in ["s", "q", "e", "g"]:
            for cdir in candidate_dirs:
                if not cdir.exists():
                    continue
                for fname in [f"rating_{r_name}.roar", f"rating_{r_name}.bin"]:
                    p = cdir / fname
                    if p.exists():
                        try:
                            with open(p, "rb") as f:
                                self.rating_masks[r_name] = BitMap.deserialize(f.read())
                            break
                        except Exception:
                            pass
                if r_name in self.rating_masks:
                    break

        # Load media masks and not_deleted mask
        for m_name in ["image", "video"]:
            for cdir in candidate_dirs:
                p = cdir / f"media_{m_name}.roar"
                if p.exists():
                    try:
                        with open(p, "rb") as f:
                            self.media_masks[m_name] = BitMap.deserialize(f.read())
                        break
                    except Exception:
                        pass

        for cdir in candidate_dirs:
            p = cdir / "not_deleted.roar"
            if p.exists():
                try:
                    with open(p, "rb") as f:
                        self.not_deleted_mask = BitMap.deserialize(f.read())
                    break
                except Exception:
                    pass

    def get_rating_mask(self, allowed_ratings: Iterable[str]) -> Optional[BitMap]:
        """Returns union bitmap of allowed ratings."""
        if BitMap is None:
            return None

        combined = BitMap()
        has_any = False
        for r in allowed_ratings:
            r_norm = r.lower().strip()
            if r_norm == "general":
                r_norm = "s"
            bm = self.rating_masks.get(r_norm)
            if bm is not None:
                combined |= bm
                has_any = True

        return combined if has_any else None

    def get_filter_mask(
        self,
        allowed_ratings: Optional[Iterable[str]] = None,
        allowed_media_types: Optional[Iterable[str]] = None,
    ) -> Optional[BitMap]:
        """
        Combines allowed ratings (S/Q/E), allowed media types (image/video),
        and strictly enforces not_deleted (never showing deleted or swf posts).
        """
        if BitMap is None:
            return None

        mask: Optional[BitMap] = None

        if allowed_ratings:
            mask = self.get_rating_mask(allowed_ratings)

        if allowed_media_types:
            media_union = BitMap()
            has_media = False
            for m in allowed_media_types:
                m_norm = "image" if m.lower().strip() in ("image", "images", "img") else "video"
                m_bm = self.media_masks.get(m_norm)
                if m_bm is not None:
                    media_union |= m_bm
                    has_media = True
            if has_media:
                mask = (mask & media_union) if mask is not None else media_union

        # Strictly enforce not_deleted (excludes deleted posts & swf)
        if self.not_deleted_mask is not None:
            mask = (mask & self.not_deleted_mask) if mask is not None else self.not_deleted_mask

        return mask

    def filter_dense_ids(
        self,
        candidate_dense_ids: List[int],
        allowed_mask: Optional[BitMap] = None,
        rejected_dense_set: Optional[Set[int]] = None,
    ) -> List[int]:
        """
        Fast hard-filtering of dense indices [0, N-1] against allowed rating mask
        and rejected dense IDs.
        """
        if not candidate_dense_ids:
            return []

        out: List[int] = []
        for didx in candidate_dense_ids:
            if rejected_dense_set is not None and didx in rejected_dense_set:
                continue
            if allowed_mask is not None and didx not in allowed_mask:
                continue
            out.append(didx)

        return out
