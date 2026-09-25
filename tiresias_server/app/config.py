from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


def get_default_data_root() -> Path:
    """
    Resolves data root path portably across Windows and Linux.
    Prioritizes TIRESIAS_DATA_ROOT environment variable,
    falling back to 'data' directory adjacent to the project root.
    """
    env_root = os.environ.get("TIRESIAS_DATA_ROOT")
    if env_root:
        return Path(env_root).resolve()
    # __file__ is in <project_root>/tiresias_server/app/config.py -> parents[2] is <project_root>
    return (Path(__file__).resolve().parents[2] / "data").resolve()


def get_default_cors_origins() -> List[str]:
    """
    Parses TIRESIAS_CORS_ORIGINS environment variable as comma-separated origins.
    Defaults to local development and extension origins.
    """
    raw = os.environ.get("TIRESIAS_CORS_ORIGINS")
    if raw is not None:
        return [o.strip() for o in raw.split(",") if o.strip()]
    return [
        "http://localhost",
        "http://127.0.0.1",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]


def get_default_cors_origin_regex() -> Optional[str]:
    raw = os.environ.get("TIRESIAS_CORS_ORIGIN_REGEX")
    if raw is not None:
        return raw if raw.strip() else None
    return r"^(moz-extension|chrome-extension)://.*$"


@dataclass
class ServerConfig:
    data_root: Path = field(default_factory=get_default_data_root)

    # Server network
    host: str = field(default_factory=lambda: os.environ.get("TIRESIAS_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(os.environ.get("TIRESIAS_PORT", "8000")))
    cors_origins: List[str] = field(default_factory=get_default_cors_origins)
    cors_origin_regex: Optional[str] = field(default_factory=get_default_cors_origin_regex)

    # Registration mode: 'open' vs 'whitelist'
    registration_mode: str = field(default_factory=lambda: os.environ.get("TIRESIAS_REGISTRATION_MODE", "open").lower())

    # Artifact paths (resolved relative to data_root)
    @property
    def mmaps_dir(self) -> Path:
        return self.data_root / "mmaps"

    @property
    def bitmaps_dir(self) -> Path:
        return self.data_root / "bitmaps"

    @property
    def features_dir(self) -> Path:
        return self.data_root / "features"

    @property
    def tags_parquet(self) -> Path:
        return self.data_root / "tags.parquet"

    @property
    def post2vec_sq8_index(self) -> Path:
        return self.features_dir / "post2vec_sq8.index"

    @property
    def faiss_index_path(self) -> Path:
        return self.post2vec_sq8_index

    @property
    def post2vec_faiss_ids(self) -> Path:
        return self.features_dir / "post2vec_faiss_ids.parquet"

    @property
    def faiss_ids_path(self) -> Path:
        return self.post2vec_faiss_ids

    @property
    def taste_centroids_npy(self) -> Path:
        return self.features_dir / "taste_centroids.npy"

    @property
    def taste_archetypes_parquet(self) -> Path:
        return self.features_dir / "taste_archetypes.parquet"

    @property
    def archetypes_parquet(self) -> Path:
        return self.taste_archetypes_parquet

    @property
    def post_cofav_parquet(self) -> Path:
        return self.features_dir / "post_cofav.parquet"

    @property
    def post_tags_parquet_dir(self) -> Path:
        return self.data_root / "post_tags_parquet"

    @property
    def server_data_dir(self) -> Path:
        """Directory for server runtime configurations and state (tiresias_server/data)."""
        return (Path(__file__).resolve().parents[1] / "data").resolve()

    @property
    def initial_suppression_json(self) -> Path:
        env_cfg = os.environ.get("TIRESIAS_SUPPRESSION_CONFIG")
        if env_cfg:
            return Path(env_cfg).resolve()
        server_cfg = self.server_data_dir / "initial_suppression.json"
        if server_cfg.exists():
            return server_cfg
        return self.data_root / "initial_suppression.json"

    custom_db_path: Optional[Path] = None

    @property
    def db_path(self) -> Path:
        if self.custom_db_path is not None:
            return self.custom_db_path.resolve()
        db_env = os.environ.get("TIRESIAS_DB_PATH")
        if db_env:
            return Path(db_env).resolve()
        return self.data_root / "tiresias_user.db"

    @property
    def telemetry_db_path(self) -> Path:
        env_val = os.environ.get("TIRESIAS_TELEMETRY_DB_PATH")
        if env_val:
            return Path(env_val).resolve()
        return self.data_root / "tiresias_telemetry.db"

    # Scoring parameters
    w_vector_sim: float = 0.50
    w_quality_log: float = 0.25
    w_collab: float = 0.15
    w_tag_match: float = 0.10
    seen_decay_factor: float = 0.30

    # Resource profile: 'performance' (desktop / server) vs 'eco' (low-spec laptop / weak VPS)
    profile: str = field(default_factory=lambda: os.environ.get("TIRESIAS_PROFILE", "performance").lower())

    # Retrieval candidate budget & execution limits (can be set explicitly or driven by profile)
    candidate_faiss_top_k: int = 300
    candidate_collab_top_k: int = 150
    candidate_popular_top_k: int = 100
    max_candidates: int = 500
    faiss_threads: int = 0  # 0 = auto/all cores, >=1 = explicit thread cap
    default_limit: int = 30
    max_limit: int = 100

    def __post_init__(self) -> None:
        is_eco = self.profile in ("eco", "low_spec", "low", "lite")
        # Apply profile defaults if not explicitly set via environment
        if is_eco:
            self.candidate_faiss_top_k = int(os.environ.get("TIRESIAS_FAISS_TOP_K", 80))
            self.candidate_collab_top_k = int(os.environ.get("TIRESIAS_COLLAB_TOP_K", 40))
            self.candidate_popular_top_k = int(os.environ.get("TIRESIAS_POPULAR_TOP_K", 30))
            self.max_candidates = int(os.environ.get("TIRESIAS_MAX_CANDIDATES", 150))
            self.faiss_threads = int(os.environ.get("TIRESIAS_FAISS_THREADS", 2))
            self.default_limit = int(os.environ.get("TIRESIAS_DEFAULT_LIMIT", 20))
            self.max_limit = int(os.environ.get("TIRESIAS_MAX_LIMIT", 50))
        else:
            self.candidate_faiss_top_k = int(os.environ.get("TIRESIAS_FAISS_TOP_K", self.candidate_faiss_top_k))
            self.candidate_collab_top_k = int(os.environ.get("TIRESIAS_COLLAB_TOP_K", self.candidate_collab_top_k))
            self.candidate_popular_top_k = int(os.environ.get("TIRESIAS_POPULAR_TOP_K", self.candidate_popular_top_k))
            self.max_candidates = int(os.environ.get("TIRESIAS_MAX_CANDIDATES", self.max_candidates))
            self.faiss_threads = int(os.environ.get("TIRESIAS_FAISS_THREADS", self.faiss_threads))
            self.default_limit = int(os.environ.get("TIRESIAS_DEFAULT_LIMIT", self.default_limit))
            self.max_limit = int(os.environ.get("TIRESIAS_MAX_LIMIT", self.max_limit))

    # Rating code mapping: s (safe/general)=0, q (questionable)=1, e (explicit)=2
    rating_map: Dict[str, int] = field(default_factory=lambda: {
        "s": 0, "g": 0, "safe": 0, "general": 0,
        "q": 1, "questionable": 1,
        "e": 2, "explicit": 2,
    })
    rating_inv_map: Dict[int, str] = field(default_factory=lambda: {
        0: "s",
        1: "q",
        2: "e",
    })
