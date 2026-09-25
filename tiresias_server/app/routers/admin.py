from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import psutil
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from ..core.auth import UserRole, require_role
from ..schemas.admin import (
    AdminInviteCreateRequest,
    AdminInviteItem,
    AdminMaintenanceRequest,
    AdminPasswordResetRequest,
    AdminRoleUpdateRequest,
    AdminUserItem,
    AdminUserListResponse,
    DatabaseFileSizeStats,
    DiskSpaceStats,
    HostCpuStats,
    HostMemoryStats,
    ProcessMemoryStats,
    SystemStatsResponse,
)

router = APIRouter(
    prefix="/api/v1/admin",
    tags=["admin"],
    dependencies=[Depends(require_role(UserRole.ADMIN))],
)


def get_engine(request: Request):
    """Retrieves the serving engine singleton from app state."""
    return request.app.state.engine


# -----------------------------------------------------------------------------
# User Management Endpoints
# -----------------------------------------------------------------------------


@router.get("/users", response_model=AdminUserListResponse)
async def list_users(
    request: Request,
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(default=50, ge=1, le=200, description="Items per page"),
    search: Optional[str] = Query(default=None, description="Search filter for user_id, username, or display_name"),
    role: Optional[int] = Query(default=None, description="Exact role filter"),
) -> AdminUserListResponse:
    """Paginated user listing with optional search and role filtering."""
    engine = get_engine(request)
    users, total = engine.db.list_users(
        page=page,
        page_size=page_size,
        search=search,
        role=role,
    )
    return AdminUserListResponse(
        total=total,
        page=page,
        page_size=page_size,
        users=[AdminUserItem(**u) for u in users],
    )


@router.post("/users/{user_id}/role")
async def update_user_role(
    user_id: str,
    payload: AdminRoleUpdateRequest,
    request: Request,
) -> Dict[str, Any]:
    """Updates user authorization role level."""
    engine = get_engine(request)
    account = engine.db.get_user_account_info(user_id)
    if not account:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User '{user_id}' not found.",
        )
    ok = engine.db.set_user_role(user_id, payload.role)
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update role for user '{user_id}'.",
        )
    return {"status": "ok", "user_id": user_id, "role": payload.role}


@router.post("/users/{user_id}/password")
async def reset_user_password(
    user_id: str,
    payload: AdminPasswordResetRequest,
    request: Request,
) -> Dict[str, Any]:
    """Resets user password."""
    engine = get_engine(request)
    account = engine.db.get_user_account_info(user_id)
    if not account:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User '{user_id}' not found.",
        )
    try:
        ok = engine.db.set_user_password(user_id, payload.password)
        if not ok:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to reset password for user '{user_id}'.",
            )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    return {"status": "ok", "user_id": user_id, "message": "Password updated successfully"}


@router.delete("/users/{user_id}")
async def delete_user(
    user_id: str,
    request: Request,
) -> Dict[str, Any]:
    """Permanently deletes a user and cascades all associated profile data."""
    engine = get_engine(request)
    ok = engine.db.delete_user(user_id)
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User '{user_id}' not found.",
        )
    return {
        "status": "ok",
        "user_id": user_id,
        "message": f"User '{user_id}' and all associated records deleted successfully.",
    }


# -----------------------------------------------------------------------------
# Invite Whitelist Endpoints (Closed Beta)
# -----------------------------------------------------------------------------


@router.get("/invites", response_model=List[AdminInviteItem])
async def list_invites(request: Request) -> List[AdminInviteItem]:
    """Lists all invited users currently on the whitelist."""
    engine = get_engine(request)
    invites = engine.db.list_invited_users()
    return [AdminInviteItem(**i) for i in invites]


@router.post("/invites")
async def add_invite(
    payload: AdminInviteCreateRequest,
    request: Request,
) -> Dict[str, Any]:
    """Adds or updates a site user on the allowed invite whitelist."""
    engine = get_engine(request)
    ok = engine.db.add_invited_user(
        site_user_id=payload.site_user_id,
        username=payload.username or "",
        note=payload.note or "",
    )
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add site user {payload.site_user_id} to whitelist.",
        )
    return {
        "status": "ok",
        "site_user_id": payload.site_user_id,
        "username": payload.username,
        "note": payload.note,
        "message": f"Site user {payload.site_user_id} successfully added to whitelist.",
    }


@router.delete("/invites/{site_user_id}")
async def remove_invite(
    site_user_id: int,
    request: Request,
) -> Dict[str, Any]:
    """Removes a user from the invite whitelist."""
    engine = get_engine(request)
    ok = engine.db.remove_invited_user(site_user_id)
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Invite for site_user_id {site_user_id} not found on whitelist.",
        )
    return {
        "status": "ok",
        "site_user_id": site_user_id,
        "message": f"Site user {site_user_id} removed from whitelist.",
    }


# -----------------------------------------------------------------------------
# System Telemetry & Operations Endpoints
# -----------------------------------------------------------------------------


@router.get("/system/stats", response_model=SystemStatsResponse)
async def get_system_stats(request: Request) -> SystemStatsResponse:
    """Returns comprehensive host telemetry, storage, database metrics, and engine status."""
    engine = get_engine(request)
    engine_status = engine.get_status()

    # Host CPU telemetry
    cpu_pct = float(psutil.cpu_percent(interval=None))
    cpu_cnt = int(psutil.cpu_count(logical=True) or 1)

    # Host RAM telemetry
    vmem = psutil.virtual_memory()
    mem_total_mb = round(vmem.total / (1024 * 1024), 2)
    mem_avail_mb = round(vmem.available / (1024 * 1024), 2)
    mem_used_mb = round(vmem.used / (1024 * 1024), 2)
    mem_pct = float(vmem.percent)

    # Process RAM RSS
    proc = psutil.Process()
    proc_rss_mb = round(proc.memory_info().rss / (1024 * 1024), 2)

    # Disk telemetry on the data directory
    db_path = Path(engine.config.db_path)
    disk_target = str(db_path.parent.resolve())
    try:
        disk_usage = psutil.disk_usage(disk_target)
        disk_total_gb = round(disk_usage.total / (1024 ** 3), 2)
        disk_free_gb = round(disk_usage.free / (1024 ** 3), 2)
        disk_used_gb = round(disk_usage.used / (1024 ** 3), 2)
        disk_pct = float(disk_usage.percent)
    except Exception:
        disk_total_gb, disk_free_gb, disk_used_gb, disk_pct = 0.0, 0.0, 0.0, 0.0

    # SQLite DB file sizes in MB
    user_db_size = round(db_path.stat().st_size / (1024 * 1024), 3) if db_path.exists() else 0.0

    telemetry_db_path = getattr(engine.config, "telemetry_db_path", None)
    if not telemetry_db_path:
        telemetry_db_path = db_path.parent / "tiresias_telemetry.db"
    else:
        telemetry_db_path = Path(telemetry_db_path)
    telem_db_size = round(telemetry_db_path.stat().st_size / (1024 * 1024), 3) if telemetry_db_path.exists() else 0.0

    # Database row statistics
    db_rows = engine.db.get_database_stats()

    # Engine status
    engine_info = {
        "profile": engine.config.profile,
        "uptime_seconds": round(time.time() - engine.start_time, 1),
        "status": "maintenance" if engine.is_maintenance_mode else "online",
        "maintenance": engine.is_maintenance_mode,
        "message": engine_status.get("message", "OK"),
        "total_posts_indexed": engine.mmaps.total_posts,
        "faiss_ready": engine.faiss.is_ready(),
        "faiss_total_vectors": engine.faiss.total_vectors,
        "faiss_threads": engine_status.get("faiss_threads", 4),
        "candidate_budget": engine_status.get("candidate_budget", {}),
        "collab_archetypes_loaded": len(engine.collab.archetype_posts),
        "collab_centroids_loaded": len(engine.collab.centroids) if engine.collab.centroids is not None else 0,
        "diagnostics_status": engine_status.get("diagnostics_status", "healthy"),
        "sanity_warnings": getattr(engine, "sanity_warnings", []),
    }

    return SystemStatsResponse(
        cpu_percent=cpu_pct,
        cpu_count=cpu_cnt,
        memory_total_mb=mem_total_mb,
        memory_available_mb=mem_avail_mb,
        memory_used_mb=mem_used_mb,
        memory_percent=mem_pct,
        process_rss_mb=proc_rss_mb,
        disk_total_gb=disk_total_gb,
        disk_free_gb=disk_free_gb,
        disk_used_gb=disk_used_gb,
        disk_percent=disk_pct,
        disk_path=disk_target,
        db_user_size_mb=user_db_size,
        db_telemetry_size_mb=telem_db_size,
        db_rows=db_rows,
        engine=engine_info,
        host_cpu=HostCpuStats(cpu_percent=cpu_pct, cpu_count=cpu_cnt),
        host_memory=HostMemoryStats(total_mb=mem_total_mb, available_mb=mem_avail_mb, used_mb=mem_used_mb, percent=mem_pct),
        process_memory=ProcessMemoryStats(rss_mb=proc_rss_mb),
        disk=DiskSpaceStats(path=disk_target, total_gb=disk_total_gb, free_gb=disk_free_gb, used_gb=disk_used_gb, percent=disk_pct),
        db_files=DatabaseFileSizeStats(user_db_mb=user_db_size, telemetry_db_mb=telem_db_size),
    )


@router.post("/system/maintenance")
async def set_maintenance_mode(
    payload: AdminMaintenanceRequest,
    request: Request,
) -> Dict[str, Any]:
    """Sets or clears engine maintenance mode."""
    engine = get_engine(request)
    engine.set_maintenance_mode(payload.maintenance)
    return {
        "status": "ok",
        "maintenance": payload.maintenance,
        "message": f"Maintenance mode {'enabled' if payload.maintenance else 'disabled'} successfully.",
    }


@router.post("/system/reload-artifacts")
async def admin_reload_artifacts(request: Request) -> Dict[str, Any]:
    """Hot-reloads serving artifacts (mmaps, bitmaps, FAISS, collab, suppression) without downtime."""
    engine = get_engine(request)
    return engine.reload_artifacts()
