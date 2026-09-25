from __future__ import annotations

import hashlib
import os
import secrets
from enum import IntEnum
from typing import Any, Dict, Optional, Tuple

from fastapi import Depends, HTTPException, Header, Request, status


class UserRole(IntEnum):
    """
    Extensible hierarchical user roles for Tiresias recommendation engine.
    Higher number indicates strictly more privileges.
    """
    GUEST = 0        # Temporary / unauthenticated guest
    USER = 10        # Standard user (manages own profile, reactions, boards)
    TESTER = 20      # Tester (manages own sandbox profiles, overrides archetype, views test stats)
    MODERATOR = 50   # Moderator (reserved for future public board / tag moderation)
    ADMIN = 100      # Administrator / Developer (system audit, maintenance, global operations)


def hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    """
    Hashes password using standard library PBKDF2-HMAC-SHA256 with 100,000 iterations.
    Zero external C-dependencies, FIPS-compliant and secure.
    """
    if not salt:
        salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        100000,
    )
    return dk.hex(), salt


def verify_password(password: str, password_hash: str, salt: str) -> bool:
    """Verifies a password against the stored hash and salt using constant-time comparison."""
    if not password or not password_hash or not salt:
        return False
    computed_hash, _ = hash_password(password, salt=salt)
    return secrets.compare_digest(computed_hash, password_hash)


def generate_secure_token() -> str:
    """Generates a high-entropy 64-character bearer token."""
    return secrets.token_hex(32)


def hash_token(raw_token: str) -> str:
    """Computes SHA-256 hash of token to store in SQLite without raw exposure."""
    return hashlib.sha256(raw_token.strip().encode("utf-8")).hexdigest()


_IN_MEMORY_ADMIN_KEY: Optional[str] = None


def get_admin_master_key() -> str:
    """
    Returns admin master secret from environment variable.
    If TIRESIAS_ADMIN_KEY is not set or empty, generates an ephemeral
    cryptographically secure key in-memory and logs a security warning.
    """
    global _IN_MEMORY_ADMIN_KEY
    env_key = os.environ.get("TIRESIAS_ADMIN_KEY", "").strip()
    if env_key:
        return env_key

    if _IN_MEMORY_ADMIN_KEY is None:
        _IN_MEMORY_ADMIN_KEY = secrets.token_hex(24)
        print(
            f"[SECURITY WARNING] TIRESIAS_ADMIN_KEY is not set in the environment! "
            f"Generated ephemeral master key: {_IN_MEMORY_ADMIN_KEY}",
            flush=True,
        )
    return _IN_MEMORY_ADMIN_KEY


def extract_bearer_token(
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None),
) -> Optional[str]:
    """Extracts raw token from Authorization: Bearer <token> or X-Api-Key header."""
    if authorization:
        parts = authorization.strip().split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1]
        elif len(parts) == 1:
            return parts[0]
    if x_api_key:
        return x_api_key.strip()
    return None


async def get_current_user_optional(
    request: Request,
    raw_token: Optional[str] = Depends(extract_bearer_token),
    x_admin_key: Optional[str] = Header(default=None),
) -> Optional[Dict[str, Any]]:
    """
    Resolves the active user without throwing 401 if unauthenticated.
    Supports master admin key bypass via X-Admin-Key header.
    """
    # 1. Master admin key bypass
    admin_key = get_admin_master_key()
    if x_admin_key and secrets.compare_digest(x_admin_key.strip(), admin_key):
        return {
            "user_id": "master_admin",
            "username": "admin",
            "display_name": "Master Admin",
            "role": int(UserRole.ADMIN),
            "owner_id": None,
            "site_source": "admin_key",
            "site_user_id": None,
            "has_password": True,
        }

    engine = getattr(request.app.state, "engine", None)
    if not engine or not getattr(engine, "db", None):
        return None

    # 2. Token validation
    if raw_token:
        # Check if raw_token itself is the master admin key
        if secrets.compare_digest(raw_token.strip(), admin_key):
            return {
                "user_id": "master_admin",
                "username": "admin",
                "display_name": "Master Admin",
                "role": int(UserRole.ADMIN),
                "owner_id": None,
                "site_source": "admin_key",
                "site_user_id": None,
                "has_password": True,
            }

        user = engine.db.validate_token(raw_token)
        if user:
            return user

    return None


async def get_current_user(
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional),
) -> Dict[str, Any]:
    """Strict dependency requiring authenticated user. Raises 401 Unauthorized if missing."""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please log in or refresh your token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return current_user


def require_role(min_role: UserRole):
    """
    Enforces minimum role privilege without spaghetti if-statements.
    Usage: @router.post("/admin-only", dependencies=[Depends(require_role(UserRole.ADMIN))])
    """
    async def role_checker(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
        user_role = int(user.get("role", int(UserRole.GUEST)))
        if user_role < int(min_role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Forbidden: Insufficient privileges. Required level: {min_role.name} (current: {UserRole(user_role).name}).",
            )
        return user

    return role_checker


def verify_user_access(current_user: Dict[str, Any], target_user_id: str, db: Any) -> bool:
    """
    Validates if current_user has right to read/modify data for target_user_id:
    - User matches target_user_id exactly
    - Current user is an ADMIN
    - Current user is a TESTER who owns target_user_id (target is in their sandbox)
    """
    if not current_user or not target_user_id:
        return False

    my_id = str(current_user.get("user_id", ""))
    target_id = str(target_user_id).strip()

    if my_id == target_id:
        return True

    user_role = int(current_user.get("role", int(UserRole.GUEST)))
    if user_role >= int(UserRole.ADMIN):
        return True

    # If current user is TESTER, check if target profile belongs to their sandbox
    if user_role >= int(UserRole.TESTER) and db is not None:
        target_info = db.get_user_account_info(target_id)
        if target_info and target_info.get("owner_id") == my_id:
            return True

    return False
