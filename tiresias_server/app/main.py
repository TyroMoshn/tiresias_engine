from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse


# Suppress harmless WinError 10054 noise when browsers abort keep-alive or preconnect sockets
if sys.platform == "win32":
    from asyncio.proactor_events import _ProactorBasePipeTransport

    _orig_call_connection_lost = _ProactorBasePipeTransport._call_connection_lost

    def _silence_connection_lost(self, exc):
        try:
            _orig_call_connection_lost(self, exc)
        except ConnectionResetError:
            pass
        except OSError as e:
            if getattr(e, "winerror", None) == 10054:
                pass
            else:
                raise

    _ProactorBasePipeTransport._call_connection_lost = _silence_connection_lost

from . import __version__
from .config import ServerConfig
from .core.engine import Engine
from .routers import (
    recommend_router,
    feedback_router,
    settings_router,
    system_router,
    boards_router,
    user_router,
    auth_router,
    admin_router,
)




@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize Engine and database
    cfg = ServerConfig()
    engine = Engine(cfg)
    app.state.engine = engine

    status = engine.get_status()
    print("=" * 65)
    print(f"TIRESIAS_ENGINE Serving Server v{__version__} STARTED")
    print(f"  - Database:     {cfg.db_path}")
    print(f"  - FAISS SQ8:    {'READY' if status['faiss_ready'] else 'NOT LOADED'} ({status['faiss_total_vectors']:,} vectors)")
    print(f"  - Mmaps:        {status['total_posts_indexed']:,} posts indexed")
    print(f"  - Collab:       {status['collab_archetypes_loaded']} archetypes, {status['collab_centroids_loaded']} centroids")
    print(f"  - Memory RSS:   {status['memory_rss_mb']} MB")
    print("=" * 65)

    yield

    # Shutdown
    print("TIRESIAS_ENGINE Serving Server shutting down...")


app = FastAPI(
    title="TIRESIAS_ENGINE Serving API",
    description="High-performance recommendation backend for Pinterest-like image boards.",
    version=__version__,
    lifespan=lifespan,
)

# CORS middleware for browser extensions and local web UIs
# W3C compliance: allow_credentials cannot be True if allow_origins contains wildcard '*'
_cors_cfg = ServerConfig()
_allow_credentials = "*" not in _cors_cfg.cors_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_cfg.cors_origins,
    allow_origin_regex=_cors_cfg.cors_origin_regex,
    allow_credentials=_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(auth_router)
app.include_router(recommend_router)
app.include_router(feedback_router)
app.include_router(settings_router)
app.include_router(system_router)
app.include_router(boards_router)
app.include_router(user_router)
app.include_router(admin_router)

ADMIN_TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "admin.html"


@app.get("/admin", response_class=HTMLResponse, include_in_schema=False)
@app.get("/admin/", response_class=HTMLResponse, include_in_schema=False)
async def admin_dashboard():
    """Serves the standalone single-page administrative web console."""
    if ADMIN_TEMPLATE_PATH.exists():
        content = ADMIN_TEMPLATE_PATH.read_text(encoding="utf-8")
    else:
        content = "<h1>Admin dashboard template not found</h1>"
    return HTMLResponse(content=content)


@app.get("/", tags=["root"])
async def root():
    return {
        "service": "TIRESIAS_ENGINE Serving API",
        "version": __version__,
        "docs_url": "/docs",
        "health_url": "/api/v1/system/health",
        "admin_url": "/admin",
    }



def run():
    import uvicorn
    cfg = ServerConfig()
    uvicorn.run(
        "app.main:app",
        host=cfg.host,
        port=cfg.port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    run()
