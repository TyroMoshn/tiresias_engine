from .recommend import router as recommend_router
from .feedback import router as feedback_router
from .settings import router as settings_router
from .system import router as system_router
from .boards import router as boards_router
from .user import router as user_router
from .auth import router as auth_router
from .admin import router as admin_router

__all__ = [
    "recommend_router",
    "feedback_router",
    "settings_router",
    "system_router",
    "boards_router",
    "user_router",
    "auth_router",
    "admin_router",
]


