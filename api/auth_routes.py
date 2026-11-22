"""
Login / logout / self-service account endpoints, admin user management, audit log.
"""

import csv
import io
import os
from datetime import datetime, time, timedelta
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, Response, UploadFile
from pydantic import BaseModel, Field

from api import auth
from api.limiter import limiter
from recommendation.campus_blocks import get_campus_blocks

_IST = ZoneInfo("Asia/Kolkata")
router = APIRouter(prefix="/api")
BLOCKS = get_campus_blocks()

VENDOR_DEFAULT_DAYS = 30
VENDOR_MAX_DAYS = 366


def login_rate_limit() -> str:
    return os.environ.get("LOGIN_RATE_LIMIT", "60/minute")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class LoginRequest(BaseModel):
    username: str = Field(..., max_length=64)
    password: str = Field(..., max_length=256)


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(..., max_length=256)
    new_password: str = Field(..., max_length=256)


class ProfileUpdate(BaseModel):
    home_block: Optional[str] = Field(None, description="Block the recommendations start from; empty to clear")


class UserCreate(BaseModel):
    username: str = Field(..., description="University ID: registration number, employee ID, vendor ID…")
    role: str
    full_name: str = ""
    home_block: Optional[str] = None
    department: Optional[str] = None
    expires_at: Optional[str] = Field(None, description="YYYY-MM-DD or ISO datetime. Vendors default to 30 days.")


class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    role: Optional[str] = None
    home_block: Optional[str] = None
    department: Optional[str] = None
    is_active: Optional[bool] = None
    expires_at: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bad(msg: str, status: int = 422):
    raise HTTPException(status_code=status, detail=msg)


def normalize_home_block(value: Optional[str]) -> Optional[str]:
    if value is None or str(value).strip() == "":
        return None
    canon = BLOCKS.canonical_id(value)
    if canon is None:
        _bad(f"Unknown block '{value}'. See GET /api/blocks for valid blocks.")
    return canon


def normalize_expiry(role: str, value: Optional[str], *, default_for_vendor: bool) -> Optional[str]:
    """Dates mean end of that day (IST). Vendors always expire, within a year."""
    now = auth.now_utc()
    if value is None or str(value).strip() == "":
        if role == "vendor" and default_for_vendor:
            return auth.iso(now + timedelta(days=VENDOR_DEFAULT_DAYS))
        return None
    value = str(value).strip()
    try:
        if len(value) == 10:
            d = datetime.fromisoformat(value).date()
            dt = datetime.combine(d, time(23, 59, 59), tzinfo=_IST)
        else:
            dt = datetime.fromisoformat(value)
            dt = dt if dt.tzinfo else dt.replace(tzinfo=_IST)
    except ValueError:
        _bad(f"expires_at '{value}' must be YYYY-MM-DD or an ISO datetime.")
    if dt <= now:
        _bad("expires_at must be in the future.")
    if role == "vendor" and dt > now + timedelta(days=VENDOR_MAX_DAYS):
        _bad(f"Vendor accounts can be valid for at most {VENDOR_MAX_DAYS} days.")
    return auth.iso(dt)


def _set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(auth.COOKIE_NAME, token, max_age=int(auth.session_ttl().total_seconds()),
                        httponly=True, samesite="lax", secure=auth.cookie_secure(), path="/")


def _me_payload(user: Dict[str, Any]) -> Dict[str, Any]:
    block = BLOCKS.get(user.get("home_block"))
    return {**user,
            "permissions": auth.permissions_for(user["role"]),
            "landing_page": auth.landing_page(user["role"]),
            "home_block_name": block["name"] if block else None,
            "home_block_anchor": block["anchor"] if block else None}


def _get_or_404(user_id: int) -> Dict[str, Any]:
    u = auth.get_user(user_id)
    if u is None:
        _bad("User not found.", 404)
    return u


def _guard_last_admin(target: Dict[str, Any], new_role: Optional[str] = None,
                      deactivating: bool = False, deleting: bool = False) -> None:
    if target["role"] != "admin" or not target["is_active"]:
        return
    losing_admin = deleting or deactivating or (new_role is not None and new_role != "admin")
    if losing_admin and auth.count_users("admin", active_only=True) <= 1:
        _bad("This is the last active admin account; create another admin first.", 409)


# ---------------------------------------------------------------------------
# Auth: login / logout / me
# ---------------------------------------------------------------------------

@router.post("/auth/login")
@limiter.limit(login_rate_limit)
def login(request: Request, response: Response, req: LoginRequest):
    try:
        user = auth.authenticate(req.username, req.password)
    except auth.LoginError as e:
        auth.audit("login_failed", actor_username=req.username.strip()[:64], detail={"reason": e.code}, request=request)
        raise HTTPException(status_code=423 if e.code == "locked" else 401, detail=e.message)
    token = auth.create_session(user["id"], request.headers.get("user-agent", ""))
    _set_session_cookie(response, token)
    auth.audit("login_success", actor=user, request=request)
    redirect = "/account?first=1" if user["must_change_password"] else auth.landing_page(user["role"])
    return {"user": _me_payload(user), "redirect": redirect}


@router.post("/auth/logout")
def logout(request: Request, response: Response):
    token = auth.token_from_request(request)
    user = auth.current_user(request)
    if token:
        auth.delete_session(token)
    response.delete_cookie(auth.COOKIE_NAME, path="/")
    if user:
        auth.audit("logout", actor=user, request=request)
    return {"status": "logged_out"}


@router.get("/auth/me")
def me(user=Depends(auth.require_user)):
    return _me_payload(user)


@router.patch("/auth/me")
def update_me(request: Request, req: ProfileUpdate, user=Depends(auth.require_user)):
    block = normalize_home_block(req.home_block)
    updated = auth.update_user(user["id"], home_block=block)
    auth.audit("profile_updated", actor=user, target=user["username"], detail={"home_block": block}, request=request)
    return _me_payload(updated)


@router.post("/auth/change-password")
def change_password(request: Request, req: ChangePasswordRequest, user=Depends(auth.require_user)):
    try:
        auth.change_own_password(user["id"], req.current_password, req.new_password)
    except ValueError as e:
        _bad(str(e))
    auth.revoke_sessions(user["id"], keep_token=auth.token_from_request(request))
    auth.audit("password_changed", actor=user, target=user["username"], request=request)
    if user["role"] == "admin":
        auth.clear_initial_admin_file()
    return {"status": "password_changed", "redirect": auth.landing_page(user["role"])}


# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

@router.get("/blocks")
def list_blocks(user=Depends(auth.require_user)):
    """Blocks a user can belong to, with the reference block their distances come from."""
    return BLOCKS.all()


@router.get("/roles")
def list_roles(user=Depends(auth.require("manage_users"))):
    return [{"role": r, "label": auth.ROLE_LABELS[r], "permissions": auth.permissions_for(r)} for r in auth.ROLES]


# ---------------------------------------------------------------------------
# Admin: user management
# ---------------------------------------------------------------------------

@router.get("/users")
def get_users(role: Optional[str] = Query(None), q: Optional[str] = Query(None, max_length=64),
              user=Depends(auth.require("manage_users"))):
    if role and role not in auth.ROLES:
        _bad(f"Unknown role '{role}'.")
    users = auth.list_users(role=role, q=q)
    counts = {r: 0 for r in auth.ROLES}
    for u in auth.list_users():
        counts[u["role"]] += 1
    return {"users": users, "counts": counts, "total": sum(counts.values())}


@router.post("/users", status_code=201)
def create_user(request: Request, req: UserCreate, admin=Depends(auth.require("manage_users"))):
    try:
        role = auth.validate_role(req.role)
        new, temp = auth.create_user(
            req.username, role, full_name=req.full_name, home_block=normalize_home_block(req.home_block),
            department=req.department, expires_at=normalize_expiry(role, req.expires_at, default_for_vendor=True))
    except ValueError as e:
        _bad(str(e), 409 if "already exists" in str(e) else 422)
    auth.audit("user_created", actor=admin, target=new["username"],
               detail={"role": new["role"], "home_block": new["home_block"], "expires_at": new["expires_at"]},
               request=request)
    return {"user": new, "temporary_password": temp}


@router.patch("/users/{user_id}")
def patch_user(user_id: int, request: Request, req: UserUpdate, admin=Depends(auth.require("manage_users"))):
    target = _get_or_404(user_id)
    fields = req.model_dump(exclude_unset=True)
    if "role" in fields:
        try:
            fields["role"] = auth.validate_role(fields["role"])
        except ValueError as e:
            _bad(str(e))
    if target["id"] == admin["id"]:
        if fields.get("role", "admin") != "admin":
            _bad("You cannot change your own role.", 409)
        if fields.get("is_active") is False:
            _bad("You cannot disable your own account.", 409)
    _guard_last_admin(target, new_role=fields.get("role"), deactivating=fields.get("is_active") is False)
    if "home_block" in fields:
        fields["home_block"] = normalize_home_block(fields["home_block"])
    role_after = fields.get("role", target["role"])
    if "expires_at" in fields:
        fields["expires_at"] = normalize_expiry(role_after, fields["expires_at"], default_for_vendor=True)
    elif role_after == "vendor" and not target["expires_at"]:
        fields["expires_at"] = normalize_expiry("vendor", None, default_for_vendor=True)
    updated = auth.update_user(user_id, **fields)
    auth.audit("user_updated", actor=admin, target=target["username"], detail=fields, request=request)
    return updated


@router.post("/users/{user_id}/reset-password")
def reset_user_password(user_id: int, request: Request, admin=Depends(auth.require("manage_users"))):
    target = _get_or_404(user_id)
    temp = auth.reset_password(user_id)
    auth.audit("password_reset", actor=admin, target=target["username"], request=request)
    return {"user": auth.get_user(user_id), "temporary_password": temp}


@router.post("/users/{user_id}/unlock")
def unlock_user(user_id: int, request: Request, admin=Depends(auth.require("manage_users"))):
    target = _get_or_404(user_id)
    auth.unlock_user(user_id)
    auth.audit("user_unlocked", actor=admin, target=target["username"], request=request)
    return auth.get_user(user_id)


@router.delete("/users/{user_id}")
def remove_user(user_id: int, request: Request, admin=Depends(auth.require("manage_users"))):
    target = _get_or_404(user_id)
    if target["id"] == admin["id"]:
        _bad("You cannot delete your own account.", 409)
    _guard_last_admin(target, deleting=True)
    auth.delete_user(user_id)
    auth.audit("user_deleted", actor=admin, target=target["username"], detail={"role": target["role"]}, request=request)
    return {"status": "deleted", "username": target["username"]}


_CSV_ALIASES = {
    "username": ("username", "user_id", "userid", "id", "registration_number", "reg_no", "employee_id"),
    "full_name": ("full_name", "name"),
    "role": ("role",),
    "home_block": ("home_block", "block"),
    "department": ("department", "dept", "school"),
    "expires_at": ("expires_at", "expiry", "valid_until"),
}
MAX_IMPORT_ROWS = 5000


@router.post("/users/import")
async def import_users(request: Request, file: UploadFile = File(...), admin=Depends(auth.require("manage_users"))):
    """
    Bulk-create accounts from a CSV with a header row. Columns: username, role, full_name,
    home_block, department, expires_at (aliases such as user_id / name / block are accepted).
    Each new account gets a one-time password, returned only in this response.
    """
    raw = await file.read()
    if len(raw) > 2 * 1024 * 1024:
        _bad("CSV too large (max 2 MB).", 413)
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        _bad("CSV must be UTF-8 encoded.")
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        _bad("CSV is empty or has no header row.")
    header = {h.strip().lower(): h for h in reader.fieldnames if h}
    colmap = {}
    for field, aliases in _CSV_ALIASES.items():
        for a in aliases:
            if a in header:
                colmap[field] = header[a]
                break
    if "username" not in colmap or "role" not in colmap:
        _bad("CSV needs at least 'username' and 'role' columns.")

    created: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for line_no, row in enumerate(reader, start=2):
        if line_no - 1 > MAX_IMPORT_ROWS:
            errors.append({"line": line_no, "username": None, "error": f"Stopped: more than {MAX_IMPORT_ROWS} rows."})
            break
        get = lambda f: (row.get(colmap[f]) or "").strip() if f in colmap else ""
        username = get("username")
        if not username and not any((v or "").strip() for v in row.values()):
            continue   # blank line
        try:
            role = auth.validate_role(get("role"))
            new, temp = auth.create_user(
                username, role, full_name=get("full_name"),
                home_block=normalize_home_block(get("home_block")),
                department=get("department"),
                expires_at=normalize_expiry(role, get("expires_at"), default_for_vendor=True))
            created.append({"username": new["username"], "full_name": new["full_name"], "role": new["role"],
                            "home_block": new["home_block"], "expires_at": new["expires_at"],
                            "temporary_password": temp})
        except (ValueError, HTTPException) as e:
            errors.append({"line": line_no, "username": username,
                           "error": e.detail if isinstance(e, HTTPException) else str(e)})
    auth.audit("users_imported", actor=admin, target=file.filename,
               detail={"created": len(created), "errors": len(errors)}, request=request)
    return {"created": created, "errors": errors, "created_count": len(created), "error_count": len(errors)}


# ---------------------------------------------------------------------------
# Admin: audit log
# ---------------------------------------------------------------------------

@router.get("/audit")
def get_audit(limit: int = Query(100, ge=1, le=500), offset: int = Query(0, ge=0),
              action: Optional[str] = Query(None, max_length=64), actor: Optional[str] = Query(None, max_length=64),
              user=Depends(auth.require("view_audit"))):
    return {"entries": auth.list_audit(limit=limit, offset=offset, action=action, actor=actor)}
