from __future__ import annotations

from typing import List, Optional
from fastapi import APIRouter, Query, Request
from ..schemas.api import PreferencesRequest, PreferencesResponse, TagBlacklistRequest, TagBlacklistResponse

router = APIRouter(prefix="/api/v1/settings", tags=["settings"])


def get_engine(request: Request):
    return request.app.state.engine


@router.get("/tag_blacklist", response_model=TagBlacklistResponse)
async def get_tag_blacklist(
    request: Request,
    user_id: str = Query(default="default_user"),
):
    engine = get_engine(request)
    bl_set = engine.db.get_user_tag_blacklist(user_id)
    return TagBlacklistResponse(
        user_id=user_id,
        blacklisted_tag_ids=sorted(bl_set),
    )


@router.put("/tag_blacklist", response_model=TagBlacklistResponse)
async def add_tag_blacklist(req: TagBlacklistRequest, request: Request):
    engine = get_engine(request)
    engine.db.add_tag_blacklist(req.user_id, req.tag_ids)
    bl_set = engine.db.get_user_tag_blacklist(req.user_id)
    return TagBlacklistResponse(
        user_id=req.user_id,
        blacklisted_tag_ids=sorted(bl_set),
    )


@router.delete("/tag_blacklist", response_model=TagBlacklistResponse)
async def remove_tag_blacklist(req: TagBlacklistRequest, request: Request):
    engine = get_engine(request)
    engine.db.remove_tag_blacklist(req.user_id, req.tag_ids)
    bl_set = engine.db.get_user_tag_blacklist(req.user_id)
    return TagBlacklistResponse(
        user_id=req.user_id,
        blacklisted_tag_ids=sorted(bl_set),
    )


@router.get("/preferences", response_model=PreferencesResponse)
async def get_preferences(
    request: Request,
    user_id: str = Query(default="default_user"),
):
    engine = get_engine(request)
    prefs = engine.db.get_user_settings(user_id)
    return PreferencesResponse(
        user_id=user_id,
        preferences=prefs,
    )


@router.put("/preferences", response_model=PreferencesResponse)
async def update_preferences(req: PreferencesRequest, request: Request):
    engine = get_engine(request)
    engine.db.set_user_settings(req.user_id, req.preferences)
    prefs = engine.db.get_user_settings(req.user_id)
    return PreferencesResponse(
        user_id=req.user_id,
        preferences=prefs,
    )
