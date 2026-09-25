from __future__ import annotations

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Request, Query, Depends
from ..schemas.api import HealthResponse, ClientErrorCreate, ClientErrorsResponse, ClientErrorItem
from ..core.auth import UserRole, require_role, get_current_user_optional

router = APIRouter(prefix="/api/v1/system", tags=["system"])


def get_engine(request: Request):
    return request.app.state.engine


@router.get("/health", response_model=HealthResponse)
async def get_health(request: Request):
    engine = get_engine(request)
    status_dict = engine.get_status()
    return HealthResponse(**status_dict)


@router.get("/diagnostics")
async def get_diagnostics(request: Request) -> Dict[str, Any]:
    """Returns the comprehensive diagnostic audit of the system."""
    engine = get_engine(request)
    return engine.diagnostics_report


@router.post("/diagnostics/run", dependencies=[Depends(require_role(UserRole.ADMIN))])
async def run_diagnostics(request: Request) -> Dict[str, Any]:
    """Runs a fresh diagnostics audit on demand and returns results (Admin only)."""
    engine = get_engine(request)
    return engine.run_diagnostics()


@router.post("/client-errors")
async def report_client_error(payload: ClientErrorCreate, request: Request) -> Dict[str, Any]:
    """Receives and stores browser client-side JavaScript error reports."""
    engine = get_engine(request)
    error_id = engine.telemetry_db.record_error(
        user_id=payload.user_id,
        error_type=payload.error_type,
        message=payload.message,
        stack=payload.stack,
        url=payload.url,
        source_file=payload.source_file,
        lineno=payload.lineno,
        colno=payload.colno,
        metadata=payload.metadata,
    )
    return {"status": "ok", "error_id": error_id}


@router.get("/client-errors", response_model=ClientErrorsResponse)
async def get_client_errors(request: Request, limit: int = Query(default=50, ge=1, le=200)):
    """Returns recent client-side JavaScript error reports for inspection."""
    engine = get_engine(request)
    errors = engine.telemetry_db.get_recent_errors(limit=limit)
    return ClientErrorsResponse(
        total_errors=engine.telemetry_db.get_count(),
        errors=[ClientErrorItem(**err) for err in errors],
    )


@router.delete("/client-errors", dependencies=[Depends(require_role(UserRole.ADMIN))])
async def clear_client_errors(request: Request) -> Dict[str, Any]:
    """Clears all stored client-side error reports (Admin only)."""
    engine = get_engine(request)
    cleared = engine.telemetry_db.clear_errors()
    return {"status": "ok", "cleared_count": cleared}


@router.get("/stats")
async def get_stats(request: Request) -> Dict[str, Any]:
    engine = get_engine(request)
    return engine.get_status()


@router.post("/maintenance/enable", dependencies=[Depends(require_role(UserRole.ADMIN))])
async def enable_maintenance(request: Request) -> Dict[str, Any]:
    engine = get_engine(request)
    engine.set_maintenance_mode(True)
    return {"status": "ok", "maintenance": True, "message": "Maintenance mode enabled"}


@router.post("/maintenance/disable", dependencies=[Depends(require_role(UserRole.ADMIN))])
async def disable_maintenance(request: Request) -> Dict[str, Any]:
    engine = get_engine(request)
    engine.set_maintenance_mode(False)
    return {"status": "ok", "maintenance": False, "message": "Maintenance mode disabled"}


@router.get("/suppression")
async def get_suppression_status(request: Request) -> Dict[str, Any]:
    """Returns the current state and loaded rules of the cold-start tag suppression system."""
    engine = get_engine(request)
    return engine.suppression.get_status()


@router.post("/suppression/reload", dependencies=[Depends(require_role(UserRole.ADMIN))])
async def reload_suppression_rules(request: Request) -> Dict[str, Any]:
    """Hot-reloads initial_suppression.json without restarting the server (Admin only)."""
    engine = get_engine(request)
    return engine.suppression.reload(config_path=engine.config.initial_suppression_json)


@router.post("/reload-artifacts", dependencies=[Depends(require_role(UserRole.ADMIN))])
async def reload_artifacts(request: Request) -> Dict[str, Any]:
    """Hot-reloads serving artifacts (mmaps, bitmaps, FAISS, collab, suppression) without downtime (Admin only)."""
    engine = get_engine(request)
    return engine.reload_artifacts()


@router.get("/archetypes")
async def get_archetypes_list(request: Request) -> Dict[str, Any]:
    """Returns overview list of all 64 collaborative taste archetypes."""
    engine = get_engine(request)
    archetypes = [
        {"id": aid, "posts_count": len(posts)}
        for aid, posts in sorted(engine.collab.archetype_posts.items())
    ]
    return {
        "total_archetypes": len(archetypes),
        "archetypes": archetypes,
    }


@router.get("/archetypes/{archetype_id}")
async def get_archetype_detail(
    archetype_id: int,
    request: Request,
    limit: int = Query(default=100, ge=1, le=500),
) -> Dict[str, Any]:
    """Returns posts and rating distribution for a specific taste archetype."""
    engine = get_engine(request)
    posts = engine.collab.get_archetype_posts(archetype_id, limit=limit)
    items = []
    rating_counts = {"s": 0, "q": 0, "e": 0}
    for pid in posts:
        didx = engine.mmaps.find_dense_idx(pid)
        meta = engine.mmaps.get_metadata(didx) if didx is not None else {}
        r = meta.get("rating", "s")
        if r in rating_counts:
            rating_counts[r] += 1
        items.append({
            "post_id": pid,
            "rating": r,
            "score": meta.get("score", 0),
            "fav_count": meta.get("fav_count", 0),
            "width": meta.get("width", 0),
            "height": meta.get("height", 0),
            "file_ext": meta.get("file_ext", "png"),
            "is_video": meta.get("is_video", False),
            "url": f"https://e621.net/posts/{pid}",
        })
    return {
        "archetype_id": archetype_id,
        "total_posts": len(items),
        "rating_distribution": rating_counts,
        "items": items,
    }

