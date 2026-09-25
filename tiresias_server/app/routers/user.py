from __future__ import annotations

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/v1/user", tags=["user"])


class UserResetRequest(BaseModel):
    user_id: Optional[str] = None
    user_ids: Optional[List[str]] = None
    reset_all: bool = False


class UserResetResponse(BaseModel):
    success: bool = True
    user_id: Optional[str] = None
    users_affected: int = 0
    feedback_deleted: int = 0
    seen_deleted: int = 0
    tag_likes_deleted: int = 0
    message: str = "User history successfully cleared."


from app.schemas.api import ArchetypeOverrideRequest, ArchetypeOverrideResponse
from app.core.auth import UserRole, get_current_user, get_current_user_optional, verify_user_access


def get_engine(request: Request):
    return request.app.state.engine


@router.post("/reset", response_model=UserResetResponse)
async def reset_user_history(
    req: UserResetRequest,
    request: Request,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional),
):
    """
    Clears feedback, seen items, and tag likes for a user, list of users, or all users.
    Enforces RBAC: reset_all requires ADMIN. Resetting specific user requires ownership or ADMIN.
    """
    engine = get_engine(request)

    # 1. Global DB reset requires ADMIN role
    if req.reset_all:
        if not current_user or int(current_user.get("role", 0)) < int(UserRole.ADMIN):
            raise HTTPException(
                status_code=403,
                detail="Global database reset is restricted to developers/administrators.",
            )

    # 2. Targeted reset requires user ownership or TESTER sandbox ownership or ADMIN
    targets = req.user_ids if req.user_ids else ([req.user_id] if req.user_id else [])
    if current_user:
        for t in targets:
            if not verify_user_access(current_user, t, engine.db):
                raise HTTPException(
                    status_code=403,
                    detail=f"You do not have permission to reset profile '{t}'. Access forbidden.",
                )

    res = engine.db.reset_user_history(
        user_id=req.user_id,
        user_ids=req.user_ids,
        reset_all=req.reset_all,
    )
    if req.reset_all:
        msg = f"History of ALL profiles ({res.get('users_affected', 0)} accounts) completely cleared."
    elif req.user_ids:
        msg = f"History of selected profiles ({len(req.user_ids)}) successfully cleared."
    else:
        msg = f"History of profile '{req.user_id}' successfully cleared. Cold start active."

    return UserResetResponse(
        success=True,
        user_id=req.user_id,
        users_affected=res.get("users_affected", 1 if req.user_id else 0),
        feedback_deleted=res.get("feedback_deleted", 0),
        seen_deleted=res.get("seen_deleted", 0),
        tag_likes_deleted=res.get("tag_likes_deleted", 0),
        message=msg,
    )


@router.post("/archetype/override", response_model=ArchetypeOverrideResponse)
async def override_user_archetype(
    req: ArchetypeOverrideRequest,
    request: Request,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional),
):
    """
    Sets or clears manual archetype lock and forced influence weight for tester mode.
    Enforces RBAC: requires TESTER or ADMIN role.
    """
    engine = get_engine(request)
    user_id = req.user_id.strip() if req.user_id else "default_user"

    if current_user:
        user_role = int(current_user.get("role", int(UserRole.GUEST)))
        if user_role < int(UserRole.TESTER):
            raise HTTPException(
                status_code=403,
                detail="Manual archetype control is available only in tester or administrator mode.",
            )
        if not verify_user_access(current_user, user_id, engine.db):
            raise HTTPException(
                status_code=403,
                detail="Tester can only override archetypes for their own sandbox profiles.",
            )

    locked_arch = req.locked_archetype
    if locked_arch is not None:
        if locked_arch < 0 or locked_arch > 63:
            raise HTTPException(status_code=400, detail="Archetype ID must be in range 0 to 63.")

    forced_w = req.forced_weight
    if forced_w is not None:
        forced_w = float(max(0.0, min(1.0, forced_w)))

    engine.db.set_user_archetype_override(user_id, locked_archetype=locked_arch, forced_weight=forced_w)

    arch_state = engine.get_user_archetype_state(user_id)

    msg_parts = []
    if locked_arch is not None:
        msg_parts.append(f"Archetype locked to #{locked_arch}")
    else:
        msg_parts.append("Archetype in auto mode")

    if forced_w is not None:
        msg_parts.append(f"Influence weight locked to {round(forced_w * 100)}%")
    else:
        msg_parts.append("Influence weight is computed adaptively")

    return ArchetypeOverrideResponse(
        success=True,
        user_id=user_id,
        taste_archetype_id=arch_state["taste_archetype_id"],
        locked_archetype=arch_state["locked_archetype"],
        forced_weight=arch_state["forced_archetype_weight"],
        effective_archetype_id=arch_state["effective_archetype_id"],
        taste_coherence=arch_state["taste_coherence"],
        archetype_influence=arch_state["archetype_influence"],
        message=f"{', '.join(msg_parts)}.",
    )


@router.get("/{user_id}/profile")
async def get_user_profile(
    user_id: str,
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Returns user profile stats: archetype, interaction counts, rich motifs, artists,
    characters, rating distribution, and engagement metrics.
    """
    engine = get_engine(request)
    if not verify_user_access(current_user, user_id, engine.db):
        raise HTTPException(
            status_code=403,
            detail="Access to another user's profile or feedback history is forbidden.",
        )
    stats = engine.db.get_user_profile_stats(user_id)

    # Attach dynamic archetype state & decay metrics
    arch_state = engine.get_user_archetype_state(user_id)
    stats["taste_archetype_id"] = arch_state["taste_archetype_id"]
    stats["locked_archetype"] = arch_state["locked_archetype"]
    stats["forced_archetype_weight"] = arch_state["forced_archetype_weight"]
    stats["effective_archetype_id"] = arch_state["effective_archetype_id"]
    stats["taste_coherence"] = arch_state["taste_coherence"]
    stats["archetype_influence"] = arch_state["archetype_influence"]
    stats["auto_archetype_influence"] = arch_state["auto_archetype_influence"]

    # Attach suppression rules status for this user
    user_tag_likes = engine.db.get_user_tag_likes(user_id)
    eff_weights = engine.suppression.get_user_effective_weights(user_tag_likes)
    stats["suppression"] = {
        "active_suppressed_tags": eff_weights,
        "likes_to_unsuppress": engine.suppression.likes_to_unsuppress,
    }

    # Enrich with user-friendly motifs, ratings distribution, and taste breakdown
    liked_pids: List[int] = []
    if engine.db._conn:
        cur = engine.db._conn.cursor()
        cur.execute(
            "SELECT post_id FROM user_feedback WHERE user_id = ? AND signal_type = 'like' ORDER BY id DESC LIMIT 100",
            (user_id,),
        )
        liked_pids = [int(r["post_id"]) for r in cur.fetchall()]

    salient_items = engine.boards.extract_salient_tags(liked_pids, top_k=30) if liked_pids else []
    # e621 tag categories: 0=general, 1=artist, 3=copyright, 4=character, 5=species, 7=meta, 8=lore
    top_motifs = [s for s in salient_items if s.get("category", 0) in (0, 3, 5, 7, 8)][:14]
    top_artists = [s for s in salient_items if s.get("category") == 1][:8]
    top_characters = [s for s in salient_items if s.get("category") in (4, 5)][:8]

    # Calculate ratings breakdown & score statistics of liked posts
    rating_counts = {"s": 0, "q": 0, "e": 0}
    scores_list: List[int] = []
    for pid in liked_pids:
        didx = engine.mmaps.find_dense_idx(pid)
        if didx is not None:
            meta = engine.mmaps.get_metadata(didx)
            r = (meta.get("rating") or "s").lower()
            if r in rating_counts:
                rating_counts[r] += 1
            scores_list.append(meta.get("score", 0))

    total_liked_rated = max(1, sum(rating_counts.values()))
    stats["rating_distribution"] = {
        "s": rating_counts["s"],
        "q": rating_counts["q"],
        "e": rating_counts["e"],
        "s_pct": round(rating_counts["s"] / total_liked_rated * 100),
        "q_pct": round(rating_counts["q"] / total_liked_rated * 100),
        "e_pct": round(rating_counts["e"] / total_liked_rated * 100),
    }
    stats["avg_liked_score"] = round(sum(scores_list) / len(scores_list)) if scores_list else 0
    likes_cnt = stats.get("likes_count", 0)
    hides_cnt = stats.get("hides_count", 0)
    stats["approval_rate"] = round(likes_cnt / max(1, likes_cnt + hides_cnt) * 100)
    stats["top_motifs"] = top_motifs
    stats["top_artists"] = top_artists
    stats["top_characters"] = top_characters
    stats["top_tag_likes"] = salient_items[:15]
    return stats


@router.get("/{user_id}/feedback")
async def get_user_feedback(
    user_id: str,
    request: Request,
    signal_type: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Returns paginated user feedback history (likes, dislikes) enriched with mmaps post metadata.
    """
    engine = get_engine(request)
    if not verify_user_access(current_user, user_id, engine.db):
        raise HTTPException(
            status_code=403,
            detail="Access to another user's profile or feedback history is forbidden.",
        )
    limit = max(1, min(200, limit))
    offset = max(0, offset)

    history = engine.db.get_user_feedback_history(
        user_id=user_id,
        signal_type=signal_type,
        limit=limit,
        offset=offset,
    )
    total_count = engine.db.get_user_feedback_count(
        user_id=user_id,
        signal_type=signal_type,
    )

    enriched_items = []
    for entry in history:
        pid = entry["post_id"]
        item = {
            "post_id": pid,
            "signal_type": entry["signal_type"],
            "created_at": entry["created_at"],
            "score": 0,
            "fav_count": 0,
            "rating": "s",
            "file_ext": "png",
            "width": 0,
            "height": 0,
            "is_video": False,
        }
        dense_idx = engine.mmaps.find_dense_idx(pid)
        if dense_idx is not None:
            meta = engine.mmaps.get_metadata(dense_idx)
            item.update({
                "score": meta.get("score", 0),
                "fav_count": meta.get("fav_count", 0),
                "rating": meta.get("rating", "s"),
                "file_ext": meta.get("file_ext", "png"),
                "width": meta.get("width", 0),
                "height": meta.get("height", 0),
                "is_video": meta.get("is_video", False),
            })
        enriched_items.append(item)

    return {
        "user_id": user_id,
        "signal_type": signal_type,
        "total": total_count,
        "limit": limit,
        "offset": offset,
        "items": enriched_items,
    }

