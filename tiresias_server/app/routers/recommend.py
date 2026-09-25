from __future__ import annotations

from typing import List, Optional
from fastapi import APIRouter, Depends, Query, Request
from ..schemas.api import FeedRequest, FeedResponse, SimilarRequest, SimilarResponse

router = APIRouter(prefix="/api/v1/recommend", tags=["recommend"])


def get_engine(request: Request):
    return request.app.state.engine


@router.post("/feed", response_model=FeedResponse)
async def get_feed_post(req: FeedRequest, request: Request):
    engine = get_engine(request)
    result = engine.recommend_feed(
        user_id=req.user_id,
        ratings=req.ratings,
        media_types=req.media_types,
        limit=req.limit,
        cursor=req.cursor,
        min_score=req.min_score,
        exclude_tags=req.exclude_tags,
    )
    return FeedResponse(**result)


@router.get("/feed", response_model=FeedResponse)
async def get_feed_get(
    request: Request,
    user_id: str = Query(default="default_user"),
    limit: int = Query(default=30, ge=1, le=100),
    cursor: int = Query(default=0, ge=0),
    min_score: int = Query(default=0),
    ratings: Optional[List[str]] = Query(default=None),
    media_types: Optional[List[str]] = Query(default=None),
):
    """GET version of /feed for easy testing in browser address bar."""
    engine = get_engine(request)
    result = engine.recommend_feed(
        user_id=user_id,
        ratings=ratings,
        media_types=media_types,
        limit=limit,
        cursor=cursor,
        min_score=min_score,
    )
    return FeedResponse(**result)


@router.post("/similar", response_model=SimilarResponse)
async def get_similar_post(req: SimilarRequest, request: Request):
    engine = get_engine(request)
    result = engine.recommend_similar(
        post_id=req.post_id,
        ratings=req.ratings,
        media_types=req.media_types,
        limit=req.limit,
    )
    return SimilarResponse(**result)


@router.get("/similar", response_model=SimilarResponse)
async def get_similar_get(
    request: Request,
    post_id: int = Query(..., description="Target post ID to find similar posts for"),
    limit: int = Query(default=20, ge=1, le=50),
    ratings: Optional[List[str]] = Query(default=None),
):
    """GET version of /similar for easy testing in browser address bar."""
    engine = get_engine(request)
    result = engine.recommend_similar(
        post_id=post_id,
        ratings=ratings,
        limit=limit,
    )
    return SimilarResponse(**result)

