from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, model_validator


class FeedItem(BaseModel):
    post_id: int
    id: Optional[int] = None
    score: float
    score_val: int = 0
    fav_count: int = 0
    rating: str = "s"
    reasons: List[str] = Field(default_factory=list)
    width: int = 0
    height: int = 0
    file_ext: Optional[str] = "png"
    is_video: bool = False

    @model_validator(mode="before")
    @classmethod
    def set_id_alias(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "id" not in data and "post_id" in data:
                data["id"] = data["post_id"]
            elif "post_id" not in data and "id" in data:
                data["post_id"] = data["id"]
        return data



class FeedRequest(BaseModel):
    user_id: str = "default_user"
    ratings: List[str] = Field(default_factory=lambda: ["s", "q"])
    media_types: Optional[List[str]] = Field(default_factory=lambda: ["image", "video"])
    limit: int = Field(default=30, ge=1, le=100)
    cursor: int = Field(default=0, ge=0)
    min_score: int = Field(default=0)
    exclude_tags: Optional[List[int]] = None


class FeedResponse(BaseModel):
    items: List[FeedItem]
    total_candidates: int
    filtered_pool_size: int
    returned_count: int
    cursor: int
    has_more: bool
    latency_ms: float


class SimilarItem(BaseModel):
    post_id: int
    id: Optional[int] = None
    similarity: float
    score: int = 0
    fav_count: int = 0
    rating: str = "s"
    width: int = 0
    height: int = 0
    file_ext: Optional[str] = "png"
    is_video: bool = False

    @model_validator(mode="before")
    @classmethod
    def set_id_alias(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "id" not in data and "post_id" in data:
                data["id"] = data["post_id"]
            elif "post_id" not in data and "id" in data:
                data["post_id"] = data["id"]
        return data


class SimilarRequest(BaseModel):
    post_id: int
    ratings: List[str] = Field(default_factory=lambda: ["s", "q"])
    media_types: Optional[List[str]] = Field(default_factory=lambda: ["image", "video"])
    limit: int = Field(default=20, ge=1, le=50)


class SimilarResponse(BaseModel):
    query_post_id: int
    items: List[SimilarItem]
    returned_count: int
    latency_ms: float


class FeedbackRequest(BaseModel):
    user_id: str = "default_user"
    post_id: int
    signal_type: str = Field(
        default="like",
        description="Signal type: 'like', 'dislike', 'hide', 'skip', 'save', etc.",
    )

    @model_validator(mode="before")
    @classmethod
    def set_signal_type_alias(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "action" in data and ("signal_type" not in data or not data.get("signal_type")):
                data["signal_type"] = data["action"]
        return data


class FeedbackResponse(BaseModel):
    success: bool = True
    user_id: str
    post_id: int
    signal_type: str


class SeenBatchRequest(BaseModel):
    user_id: str = "default_user"
    post_ids: List[int]


class SeenBatchResponse(BaseModel):
    success: bool = True
    recorded_count: int


class TagBlacklistRequest(BaseModel):
    user_id: str = "default_user"
    tag_ids: List[int]


class TagBlacklistResponse(BaseModel):
    user_id: str
    blacklisted_tag_ids: List[int]


class PreferencesRequest(BaseModel):
    user_id: str = "default_user"
    preferences: Dict[str, str]


class PreferencesResponse(BaseModel):
    user_id: str
    preferences: Dict[str, str]


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    memory_rss_mb: float
    total_posts_indexed: int
    faiss_ready: bool
    faiss_total_vectors: int
    collab_archetypes_loaded: int
    collab_centroids_loaded: int
    database_path: str
    maintenance: bool = False
    message: Optional[str] = None
    sanity_warnings: List[str] = Field(default_factory=list)
    diagnostics_status: str = "healthy"
    telemetry_errors_count: int = 0


class ClientErrorCreate(BaseModel):
    user_id: str = "default_user"
    error_type: str = "js_error"
    message: str = ""
    stack: Optional[str] = None
    url: Optional[str] = None
    source_file: Optional[str] = None
    lineno: Optional[int] = None
    colno: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class ClientErrorItem(BaseModel):
    id: int
    timestamp: str
    user_id: str
    error_type: str
    message: str
    stack: Optional[str] = None
    url: Optional[str] = None
    source_file: Optional[str] = None
    lineno: Optional[int] = None
    colno: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class ClientErrorsResponse(BaseModel):
    total_errors: int
    errors: List[ClientErrorItem]


# =============================================================================
# Board Schemas
# =============================================================================

class BoardCreateRequest(BaseModel):
    user_id: str = "default_user"
    name: Optional[str] = None
    title: Optional[str] = None
    description: str = Field(default="", max_length=500)
    cover_post_id: Optional[int] = None
    is_public: bool = False

    @model_validator(mode="before")
    @classmethod
    def set_name_title_compat(cls, data: Any) -> Any:
        if isinstance(data, dict):
            n = data.get("name") or data.get("title")
            if not n:
                raise ValueError("Field 'name' or 'title' is required to create a board")
            data["name"] = str(n)
            data["title"] = str(n)
        return data


class BoardUpdateRequest(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=100)
    description: Optional[str] = Field(default=None, max_length=500)
    cover_post_id: Optional[int] = None
    is_public: Optional[bool] = None


class BoardSummary(BaseModel):
    board_id: str
    id: Optional[str] = None
    user_id: str
    name: str
    title: Optional[str] = None
    description: str = ""
    cover_post_id: Optional[int] = None
    is_public: bool = False
    post_count: int = 0
    created_at: float
    updated_at: float

    @model_validator(mode="before")
    @classmethod
    def set_board_alias(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "id" not in data and "board_id" in data:
                data["id"] = str(data["board_id"])
            if "title" not in data and "name" in data:
                data["title"] = str(data["name"])
        return data


class BoardPostItem(BaseModel):
    post_id: int
    id: Optional[int] = None
    added_at: float
    score: int = 0
    fav_count: int = 0
    epoch_day: int = 0
    rating: str = "s"
    preview_url: Optional[str] = None
    width: int = 0
    height: int = 0
    affinity: float = 0.0

    @model_validator(mode="before")
    @classmethod
    def set_id_alias(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "id" not in data and "post_id" in data:
                data["id"] = data["post_id"]
            elif "post_id" not in data and "id" in data:
                data["post_id"] = data["id"]
        return data


class SalientTagItem(BaseModel):
    tag: str
    score: float
    tf: float
    idf: float
    category: int = 0
    post_count: int = 0


class BoardCoverage(BaseModel):
    total_posts: int = 0
    indexed_posts: int = 0
    coverage_ratio: float = 0.0
    status: str = "sufficient"
    warning_message: Optional[str] = None


class BoardDetailResponse(BaseModel):
    board_id: str
    user_id: str
    name: str
    description: str = ""
    cover_post_id: Optional[int] = None
    is_public: bool = False
    post_count: int
    created_at: float
    updated_at: float
    salient_tags: List[SalientTagItem] = Field(default_factory=list)
    posts: List[BoardPostItem] = Field(default_factory=list)
    coverage: Optional[BoardCoverage] = None
    limit: int = 50
    offset: int = 0
    sort_by: str = "added_at"
    order: str = "desc"


class BoardPostsAddRequest(BaseModel):
    post_ids: List[int] = Field(..., min_items=1)


class BoardPostsAddResponse(BaseModel):
    success: bool = True
    board_id: str
    added_count: int
    total_post_count: int


class BoardRecommendRequest(BaseModel):
    user_id: str = "default_user"
    ratings: Optional[List[str]] = None
    limit: int = Field(default=30, ge=1, le=1000)
    min_score: int = Field(default=0)
    exclude_tags: Optional[List[int]] = None


class BoardRecommendItem(BaseModel):
    post_id: int
    score: float
    similarity: float
    score_val: int = 0
    fav_count: int = 0
    rating: str = "s"
    width: int = 0
    height: int = 0


class BoardRecommendResponse(BaseModel):
    board_id: str
    items: List[BoardRecommendItem] = Field(default_factory=list)
    total_candidates: int = 0
    filtered_pool_size: int = 0
    returned_count: int = 0
    coverage: Optional[BoardCoverage] = None
    latency_ms: float = 0.0
    message: Optional[str] = None
    error: Optional[str] = None


class BoardExportResponse(BaseModel):
    board_id: str
    name: str
    description: str = ""
    created_at: float
    updated_at: float
    cover_post_id: Optional[int] = None
    post_ids: List[int]
    exported_at: float
    engine_version: str = "1.0.0"


class BoardImportRequest(BaseModel):
    user_id: str = "default_user"
    name: Optional[str] = None
    description: Optional[str] = None
    post_ids: List[int] = Field(default_factory=list)
    cover_post_id: Optional[int] = None


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
    message: str = "History reset successfully."


class ArchetypeOverrideRequest(BaseModel):
    user_id: str = "default_user"
    locked_archetype: Optional[int] = Field(
        default=None,
        description="Lock to specific archetype ID (0-63). Set to None to enable auto migration.",
    )
    forced_weight: Optional[float] = Field(
        default=None,
        description="Forced archetype influence weight (0.0 - 1.0). Set to None to use dynamic decay.",
    )


class ArchetypeOverrideResponse(BaseModel):
    success: bool = True
    user_id: str
    taste_archetype_id: int
    locked_archetype: Optional[int] = None
    forced_weight: Optional[float] = None
    effective_archetype_id: int
    taste_coherence: float = 0.0
    archetype_influence: float = 1.0
    message: str = "Archetype settings successfully updated."


# -----------------------------------------------------------------------------
# Authentication & Accounts Schemas
# -----------------------------------------------------------------------------

class SessionHandshakeRequest(BaseModel):
    site: str = "e621"
    site_user_id: int
    username: str
    password: Optional[str] = None
    device_info: Optional[str] = ""


class LocalRegisterRequest(BaseModel):
    username: str
    password: str
    display_name: Optional[str] = None
    device_info: Optional[str] = ""


class LocalLoginRequest(BaseModel):
    username: str
    password: str
    device_info: Optional[str] = ""


class SetPasswordRequest(BaseModel):
    new_password: str
    new_username: Optional[str] = None


class LinkSessionRequest(BaseModel):
    site_user_id: int
    site_username: str


class CreateSandboxProfileRequest(BaseModel):
    profile_name: str


class AuthUserResponse(BaseModel):
    success: bool = True
    user_id: str
    username: str
    display_name: str
    role: int
    role_name: str
    site_source: str
    site_user_id: Optional[int] = None
    has_password: bool = False
    token: Optional[str] = None
    owner_id: Optional[str] = None
    sandbox_profiles: Optional[List[Dict[str, Any]]] = None
    message: Optional[str] = None

