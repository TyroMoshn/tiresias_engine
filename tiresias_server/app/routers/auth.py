from __future__ import annotations

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Request, status

from ..core.auth import (
    UserRole,
    extract_bearer_token,
    get_current_user,
    require_role,
)
from ..core.rate_limit import rate_limit_auth
from ..schemas.api import (
    AuthUserResponse,
    CreateSandboxProfileRequest,
    LocalLoginRequest,
    LocalRegisterRequest,
    SessionHandshakeRequest,
    SetPasswordRequest,
    LinkSessionRequest,
)

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


def get_engine(request: Request):
    return request.app.state.engine


@router.post(
    "/session-handshake",
    response_model=AuthUserResponse,
    dependencies=[Depends(rate_limit_auth)],
)
async def session_handshake(req: SessionHandshakeRequest, request: Request):
    """
    Seamless zero-click onboarding: registers or updates e621 user and issues a device Bearer token.
    Enforces password check on protected accounts and whitelist check in closed beta.
    """
    engine = get_engine(request)
    engine_cfg = getattr(engine, "config", getattr(engine, "cfg", None))
    reg_mode = getattr(engine_cfg, "registration_mode", "open") if engine_cfg else "open"
    try:
        user_dict, raw_token = engine.db.create_or_get_e621_user(
            site_user_id=req.site_user_id,
            username=req.username,
            device_info=req.device_info or "",
            password=req.password,
            registration_mode=reg_mode,
        )
    except ValueError as ve:
        err_msg = str(ve)
        if "PASSWORD_REQUIRED" in err_msg:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=err_msg)
        if "REGISTRATION_CLOSED" in err_msg:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=err_msg)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=err_msg)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    role_val = int(user_dict.get("role", int(UserRole.USER)))
    return AuthUserResponse(
        success=True,
        user_id=user_dict["user_id"],
        username=user_dict.get("username", req.username),
        display_name=user_dict.get("display_name", f"{req.username} (e621)"),
        role=role_val,
        role_name=UserRole(role_val).name,
        site_source="e621",
        site_user_id=req.site_user_id,
        has_password=bool(user_dict.get("has_password")),
        token=raw_token,
        owner_id=user_dict.get("owner_id"),
        message="e621 session successfully synchronized with Tiresias.",
    )


@router.post(
    "/register",
    response_model=AuthUserResponse,
    dependencies=[Depends(rate_limit_auth)],
)
async def register_local(req: LocalRegisterRequest, request: Request):
    """
    Direct registration of a standalone Tiresias account with username and password.
    Guarantees strict namespacing ('local:<username>') with no collision with e621 users.
    """
    engine = get_engine(request)
    try:
        user_dict, raw_token = engine.db.register_local_user(
            username=req.username,
            password=req.password,
            role=int(UserRole.USER),
            display_name=req.display_name,
            device_info=req.device_info or "",
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Registration failed: {e}")

    role_val = int(user_dict.get("role", int(UserRole.USER)))
    return AuthUserResponse(
        success=True,
        user_id=user_dict["user_id"],
        username=user_dict["username"],
        display_name=user_dict["display_name"],
        role=role_val,
        role_name=UserRole(role_val).name,
        site_source="direct",
        has_password=True,
        token=raw_token,
        owner_id=user_dict.get("owner_id"),
        message="Local Tiresias profile successfully created.",
    )


@router.post(
    "/login",
    response_model=AuthUserResponse,
    dependencies=[Depends(rate_limit_auth)],
)
async def login_local(req: LocalLoginRequest, request: Request):
    """
    Authenticates username and password, returning a fresh device Bearer token.
    """
    engine = get_engine(request)
    auth_res = engine.db.authenticate_local_user(
        username=req.username,
        password=req.password,
        device_info=req.device_info or "",
    )
    if not auth_res:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password.",
        )

    user_dict, raw_token = auth_res
    role_val = int(user_dict.get("role", int(UserRole.USER)))
    sandbox = []
    if role_val >= int(UserRole.TESTER):
        sandbox = engine.db.list_tester_sandbox_profiles(user_dict["user_id"])

    return AuthUserResponse(
        success=True,
        user_id=user_dict["user_id"],
        username=user_dict.get("username", req.username),
        display_name=user_dict.get("display_name", req.username),
        role=role_val,
        role_name=UserRole(role_val).name,
        site_source=user_dict.get("site_source", "direct"),
        site_user_id=user_dict.get("site_user_id"),
        has_password=True,
        token=raw_token,
        owner_id=user_dict.get("owner_id"),
        sandbox_profiles=sandbox,
        message="Login successful.",
    )


@router.post("/logout")
async def logout(
    request: Request,
    raw_token: Optional[str] = Depends(extract_bearer_token),
):
    """
    Revokes the active Bearer token from SQLite.
    """
    engine = get_engine(request)
    if raw_token:
        engine.db.revoke_token(raw_token)
    return {"success": True, "message": "Session successfully terminated."}


@router.get("/me", response_model=AuthUserResponse)
async def get_me(request: Request, current_user: Dict[str, Any] = Depends(get_current_user)):
    """
    Returns profile details, role, and sandbox profiles of the currently authenticated user.
    """
    engine = get_engine(request)
    user_id = current_user["user_id"]
    role_val = int(current_user.get("role", int(UserRole.USER)))

    sandbox = []
    if role_val >= int(UserRole.TESTER):
        sandbox = engine.db.list_tester_sandbox_profiles(user_id)

    return AuthUserResponse(
        success=True,
        user_id=user_id,
        username=current_user.get("username") or user_id,
        display_name=current_user.get("display_name") or user_id,
        role=role_val,
        role_name=UserRole(role_val).name,
        site_source=current_user.get("site_source", "direct"),
        site_user_id=current_user.get("site_user_id"),
        has_password=bool(current_user.get("has_password")),
        token=None,
        owner_id=current_user.get("owner_id"),
        sandbox_profiles=sandbox,
        message="Profile is active.",
    )


@router.post("/password")
async def set_password(
    req: SetPasswordRequest,
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Sets or updates password for the current account to allow multi-device login.
    Optionally allows setting or updating a readable username.
    """
    engine = get_engine(request)
    try:
        engine.db.set_user_password(
            user_id=current_user["user_id"],
            new_password=req.new_password,
            new_username=req.new_username,
        )
        return {"success": True, "message": "Password successfully updated."}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))


@router.post("/link-session", response_model=AuthUserResponse)
async def link_session(
    req: LinkSessionRequest,
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Links active e621 session (site_user_id and site_username) to the current user.
    """
    engine = get_engine(request)
    try:
        engine.db.link_site_session(
            user_id=current_user["user_id"],
            site_user_id=req.site_user_id,
            site_username=req.site_username,
        )
        updated = engine.db.get_user_account_info(current_user["user_id"])
        if not updated:
            raise HTTPException(status_code=404, detail="User not found.")
        role_val = int(updated.get("role", int(UserRole.USER)))
        return AuthUserResponse(
            success=True,
            user_id=updated["user_id"],
            username=updated.get("username") or updated["user_id"],
            display_name=updated.get("display_name") or updated["user_id"],
            role=role_val,
            role_name=UserRole(role_val).name,
            site_source=updated.get("site_source", "e621"),
            site_user_id=updated.get("site_user_id"),
            has_password=bool(updated.get("has_password")),
            token=None,
            owner_id=updated.get("owner_id"),
            message="Site session successfully linked to account.",
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# -----------------------------------------------------------------------------
# Tester Profile Sandbox Endpoints
# -----------------------------------------------------------------------------

@router.get("/sandbox")
async def list_sandbox_profiles(
    request: Request,
    current_user: Dict[str, Any] = Depends(require_role(UserRole.TESTER)),
):
    """Lists all sandbox sub-profiles created by this tester."""
    engine = get_engine(request)
    profiles = engine.db.list_tester_sandbox_profiles(current_user["user_id"])
    return {"success": True, "profiles": profiles}


@router.post("/sandbox/create")
async def create_sandbox_profile(
    req: CreateSandboxProfileRequest,
    request: Request,
    current_user: Dict[str, Any] = Depends(require_role(UserRole.TESTER)),
):
    """Creates a new sandbox profile for testing different tastes and cold-start."""
    engine = get_engine(request)
    try:
        profile = engine.db.create_sandbox_profile(current_user["user_id"], req.profile_name)
        return {"success": True, "profile": profile}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))


@router.delete("/sandbox/{profile_id}")
async def delete_sandbox_profile(
    profile_id: str,
    request: Request,
    current_user: Dict[str, Any] = Depends(require_role(UserRole.TESTER)),
):
    """Deletes a sandbox profile owned by this tester."""
    engine = get_engine(request)
    deleted = engine.db.delete_sandbox_profile(current_user["user_id"], profile_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Sandbox profile not found or belongs to another user.")
    return {"success": True, "deleted_id": profile_id}
