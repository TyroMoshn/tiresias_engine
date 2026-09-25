from __future__ import annotations

import datetime
import time
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel

from ..core.engine import Engine
from ..core.auth import get_current_user_optional, verify_user_access
from ..schemas.api import (
    BoardCreateRequest,
    BoardDetailResponse,
    BoardExportResponse,
    BoardImportRequest,
    BoardPostItem,
    BoardPostsAddRequest,
    BoardPostsAddResponse,
    BoardRecommendItem,
    BoardRecommendRequest,
    BoardRecommendResponse,
    BoardSummary,
    BoardUpdateRequest,
    SalientTagItem,
)

router = APIRouter(prefix="/api/v1/boards", tags=["boards"])


def get_engine(request: Request) -> Engine:
    return request.app.state.engine


# -----------------------------------------------------------------------------
# 1. Board CRUD
# -----------------------------------------------------------------------------

@router.post("", response_model=BoardSummary, status_code=201)
def create_board(
    req: BoardCreateRequest,
    engine: Engine = Depends(get_engine),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional),
) -> Dict[str, Any]:
    """Creates a new board for a user."""
    target_user_id = req.user_id
    if current_user:
        if req.user_id and req.user_id != "default_user":
            if not verify_user_access(current_user, req.user_id, engine.db):
                raise HTTPException(status_code=403, detail=f"Insufficient permissions to create board on behalf of '{req.user_id}'.")
            target_user_id = req.user_id
        else:
            target_user_id = current_user["user_id"]

    board = engine.db.create_board(
        user_id=target_user_id,
        name=req.name,
        description=req.description,
        cover_post_id=req.cover_post_id,
        is_public=req.is_public,
    )
    return board


@router.get("", response_model=List[BoardSummary])
def list_boards(
    user_id: str = Query(default="default_user"),
    engine: Engine = Depends(get_engine),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional),
) -> List[Dict[str, Any]]:
    """Lists boards belonging to a user. Private boards are only returned to their owner or admin."""
    boards = engine.db.list_user_boards(user_id=user_id)
    if current_user and verify_user_access(current_user, user_id, engine.db):
        return boards
    elif user_id == "default_user":
        return boards
    else:
        return [b for b in boards if b.get("is_public", False)]


@router.get("/{board_id}", response_model=BoardDetailResponse)
def get_board(
    board_id: str,
    sort_by: str = Query(default="added_at", description="Sort by: 'added_at', 'epoch_day', 'score', 'fav_count', 'affinity'"),
    order: str = Query(default="desc", description="Sort order: 'desc' or 'asc'"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    engine: Engine = Depends(get_engine),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional),
) -> Dict[str, Any]:
    """
    Returns complete board details, salient theme tags, and hydrated posts
    sorted according to the requested criteria (including semantic affinity to board centroid).
    Private boards require owner or admin authentication.
    """
    board = engine.db.get_board(board_id)
    if not board:
        raise HTTPException(status_code=404, detail=f"Board '{board_id}' not found")

    is_pub = board.get("is_public", False)
    if not is_pub and current_user and not verify_user_access(current_user, board["user_id"], engine.db):
        raise HTTPException(status_code=403, detail="This board is private. Access forbidden.")

    # Hydrate and sort posts
    posts, total_count = engine.boards.get_hydrated_board_posts(
        board_id=board_id,
        sort_by=sort_by.lower().strip(),
        order=order.lower().strip(),
        limit=limit,
        offset=offset,
    )

    # Extract salient tags
    all_pids = [p["post_id"] for p in posts]
    salient_tags = engine.boards.extract_salient_tags(all_pids, top_k=8)

    # Calculate model indexing coverage
    coverage = engine.boards.get_board_coverage(board_id)

    return {
        "board_id": board["board_id"],
        "user_id": board["user_id"],
        "name": board["name"],
        "description": board["description"],
        "cover_post_id": board["cover_post_id"],
        "is_public": is_pub,
        "post_count": total_count,
        "created_at": board["created_at"],
        "updated_at": board["updated_at"],
        "salient_tags": salient_tags,
        "posts": posts,
        "coverage": coverage,
        "limit": limit,
        "offset": offset,
        "sort_by": sort_by,
        "order": order,
    }


@router.patch("/{board_id}", response_model=BoardSummary)
def update_board(
    board_id: str,
    req: BoardUpdateRequest,
    engine: Engine = Depends(get_engine),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional),
) -> Dict[str, Any]:
    """Updates board title, description, cover image, or public/private visibility."""
    board = engine.db.get_board(board_id)
    if not board:
        raise HTTPException(status_code=404, detail=f"Board '{board_id}' not found")
    if current_user and not verify_user_access(current_user, board["user_id"], engine.db):
        raise HTTPException(status_code=403, detail="Forbidden: You are not the owner of this board.")

    success = engine.db.update_board(
        board_id=board_id,
        name=req.name,
        description=req.description,
        cover_post_id=req.cover_post_id,
        is_public=req.is_public,
    )
    if not success:
        raise HTTPException(status_code=404, detail=f"Board '{board_id}' not found or nothing updated")

    board = engine.db.get_board(board_id)
    return board


@router.delete("/{board_id}")
def delete_board(
    board_id: str,
    engine: Engine = Depends(get_engine),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional),
) -> Dict[str, Any]:
    """Deletes a board and cascades removal of all post associations."""
    board = engine.db.get_board(board_id)
    if not board:
        raise HTTPException(status_code=404, detail=f"Board '{board_id}' not found")
    if current_user and not verify_user_access(current_user, board["user_id"], engine.db):
        raise HTTPException(status_code=403, detail="Forbidden: You are not the owner of this board.")

    success = engine.db.delete_board(board_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Board '{board_id}' not found")
    return {"success": True, "board_id": board_id, "message": "Board successfully deleted"}


# -----------------------------------------------------------------------------
# 2. Board Posts Management
# -----------------------------------------------------------------------------

@router.post("/{board_id}/posts", response_model=BoardPostsAddResponse)
def add_posts_to_board(
    board_id: str,
    req: BoardPostsAddRequest,
    engine: Engine = Depends(get_engine),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional),
) -> Dict[str, Any]:
    """Adds one or more post IDs to the board."""
    board = engine.db.get_board(board_id)
    if not board:
        raise HTTPException(status_code=404, detail=f"Board '{board_id}' not found")
    if current_user and not verify_user_access(current_user, board["user_id"], engine.db):
        raise HTTPException(status_code=403, detail="Forbidden: You are not the owner of this board.")

    added_count = engine.db.add_posts_to_board(board_id, req.post_ids)
    total_count = engine.db.get_board_post_count(board_id)

    return {
        "success": True,
        "board_id": board_id,
        "added_count": added_count,
        "total_post_count": total_count,
    }


@router.delete("/{board_id}/posts/{post_id}")
def remove_post_from_board(
    board_id: str,
    post_id: int,
    engine: Engine = Depends(get_engine),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional),
) -> Dict[str, Any]:
    """Removes a specific post from the board."""
    board = engine.db.get_board(board_id)
    if not board:
        raise HTTPException(status_code=404, detail=f"Board '{board_id}' not found")
    if current_user and not verify_user_access(current_user, board["user_id"], engine.db):
        raise HTTPException(status_code=403, detail="Forbidden: You are not the owner of this board.")

    removed = engine.db.remove_post_from_board(board_id, post_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Post {post_id} was not present in board '{board_id}'")

    total_count = engine.db.get_board_post_count(board_id)
    return {
        "success": True,
        "board_id": board_id,
        "post_id": post_id,
        "total_post_count": total_count,
    }


# -----------------------------------------------------------------------------
# 3. 'More Like This Board' Recommendations
# -----------------------------------------------------------------------------

@router.post("/{board_id}/recommend", response_model=BoardRecommendResponse)
def recommend_for_board_post(
    board_id: str,
    req: BoardRecommendRequest,
    engine: Engine = Depends(get_engine),
) -> Dict[str, Any]:
    """
    Generates recommendations matching the collective aesthetic of the board (POST method).
    Guarantees hard exclusion of all posts already in the board.
    """
    result = engine.boards.recommend_for_board(
        board_id=board_id,
        user_id=req.user_id,
        ratings=req.ratings,
        limit=req.limit,
        min_score=req.min_score,
        exclude_tags=req.exclude_tags,
    )
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@router.get("/{board_id}/recommend", response_model=BoardRecommendResponse)
def recommend_for_board_get(
    board_id: str,
    user_id: str = Query(default="default_user"),
    ratings: Optional[str] = Query(default="s,q", description="Comma-separated list, e.g. 's,q'"),
    limit: int = Query(default=30, ge=1, le=1000),
    min_score: int = Query(default=0),
    engine: Engine = Depends(get_engine),
) -> Dict[str, Any]:
    """
    Generates recommendations matching the collective aesthetic of the board (GET method).
    Convenient for direct browser URL navigation and testing.
    """
    rating_list = [r.strip().lower() for r in ratings.split(",") if r.strip()] if ratings else ["s", "q"]
    result = engine.boards.recommend_for_board(
        board_id=board_id,
        user_id=user_id,
        ratings=rating_list,
        limit=limit,
        min_score=min_score,
    )
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


# -----------------------------------------------------------------------------
# 4. Board Export & Import
# -----------------------------------------------------------------------------

@router.get("/{board_id}/export", response_model=BoardExportResponse)
def export_board(
    board_id: str,
    engine: Engine = Depends(get_engine),
) -> Dict[str, Any]:
    """Exports board metadata and post IDs as a standalone portable JSON."""
    board = engine.db.get_board(board_id)
    if not board:
        raise HTTPException(status_code=404, detail=f"Board '{board_id}' not found")

    post_records = engine.db.get_board_post_records(board_id)
    post_ids = [r["post_id"] for r in post_records]

    return {
        "board_id": board["board_id"],
        "name": board["name"],
        "description": board["description"],
        "created_at": board["created_at"],
        "updated_at": board["updated_at"],
        "cover_post_id": board["cover_post_id"],
        "post_ids": post_ids,
        "exported_at": time.time(),
        "engine_version": "1.0.0",
    }


@router.post("/import", response_model=BoardSummary, status_code=201)
def import_board(
    req: BoardImportRequest,
    engine: Engine = Depends(get_engine),
) -> Dict[str, Any]:
    """Imports a board from a JSON payload into the specified user's profile."""
    board_name = req.name or f"Imported Board ({datetime.date.today()})"
    desc = req.description or "Imported collection"
    new_board = engine.db.create_board(
        user_id=req.user_id,
        name=board_name,
        description=desc,
        cover_post_id=req.cover_post_id,
    )
    bid = new_board["board_id"]
    if req.post_ids:
        engine.db.add_posts_to_board(bid, req.post_ids)

    # Return refreshed board with accurate post_count
    imported = engine.db.get_board(bid)
    return imported or new_board
