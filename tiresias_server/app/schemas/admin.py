from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class AdminUserItem(BaseModel):
    user_id: str
    username: Optional[str] = None
    display_name: Optional[str] = None
    role: int = 10
    site_source: Optional[str] = "direct"
    site_user_id: Optional[int] = None
    owner_id: Optional[str] = None
    created_at: float
    updated_at: Optional[float] = None
    taste_archetype_id: Optional[int] = 27
    locked_archetype: Optional[int] = None
    forced_archetype_weight: Optional[float] = None
    has_password: bool = False

    model_config = {"extra": "allow"}


class AdminUserListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    users: List[AdminUserItem]


class AdminRoleUpdateRequest(BaseModel):
    role: int = Field(..., ge=0, le=100, description="Role level (0=GUEST, 10=USER, 20=TESTER, 50=MODERATOR, 100=ADMIN)")


class AdminPasswordResetRequest(BaseModel):
    password: str = Field(..., min_length=4, description="New user password (minimum 4 characters)")


class AdminInviteCreateRequest(BaseModel):
    site_user_id: int = Field(..., description="Numeric user ID on e621")
    username: Optional[str] = Field(default="", description="Username on e621 (optional)")
    note: Optional[str] = Field(default="", description="Optional note or reference")


class AdminInviteItem(BaseModel):
    site_user_id: int
    username: Optional[str] = ""
    note: Optional[str] = ""
    created_at: float

    model_config = {"extra": "allow"}


class AdminMaintenanceRequest(BaseModel):
    maintenance: bool = Field(..., description="True to enable maintenance mode, False to disable")


class HostCpuStats(BaseModel):
    cpu_percent: float
    cpu_count: int


class HostMemoryStats(BaseModel):
    total_mb: float
    available_mb: float
    used_mb: float
    percent: float


class ProcessMemoryStats(BaseModel):
    rss_mb: float


class DiskSpaceStats(BaseModel):
    path: str
    total_gb: float
    free_gb: float
    used_gb: float
    percent: float


class DatabaseFileSizeStats(BaseModel):
    user_db_mb: float
    telemetry_db_mb: float


class DatabaseRowStats(BaseModel):
    users: int
    user_feedback: int
    user_boards: int
    allowed_invites: int
    auth_tokens: int


class SystemStatsResponse(BaseModel):
    # Flattened telemetry fields
    cpu_percent: float
    cpu_count: int
    memory_total_mb: float
    memory_available_mb: float
    memory_used_mb: float
    memory_percent: float
    process_rss_mb: float
    disk_total_gb: float
    disk_free_gb: float
    disk_used_gb: float
    disk_percent: float
    disk_path: str
    db_user_size_mb: float
    db_telemetry_size_mb: float
    db_rows: Dict[str, int]
    engine: Dict[str, Any]

    # Structured sub-objects
    host_cpu: Optional[HostCpuStats] = None
    host_memory: Optional[HostMemoryStats] = None
    process_memory: Optional[ProcessMemoryStats] = None
    disk: Optional[DiskSpaceStats] = None
    db_files: Optional[DatabaseFileSizeStats] = None

    model_config = {"extra": "allow"}
