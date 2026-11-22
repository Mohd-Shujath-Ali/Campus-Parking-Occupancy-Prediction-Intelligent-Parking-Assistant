"""
Authentication and role-based access control.

One login for everyone. The role on the account decides what the server allows; the UI only
hides what a role cannot use. Every protected endpoint declares its permission with
`Depends(require("<permission>"))`, so the check always happens on the server.

- Passwords: scrypt (stdlib hashlib), per-user random salt, constant-time comparison.
- Sessions: random 256-bit token in an HttpOnly cookie; only its SHA-256 is stored, so a
  leaked database cannot be replayed as a login. Logout and password resets revoke sessions.
- Lockout: N consecutive failures lock the account for a few minutes.
- Accounts can expire (used for vendors); expired accounts cannot log in and lose sessions.
- Accounts created by an admin get a one-time password that must be changed at first login.
"""

import base64
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from fastapi import HTTPException, Request

from api.db import get_db

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Roles and permissions
# ---------------------------------------------------------------------------

ROLES = ("admin", "warden", "faculty", "student", "vendor", "guard")

ROLE_LABELS = {
    "admin": "Administrator",
    "warden": "Warden",
    "faculty": "Faculty",
    "student": "Student",
    "vendor": "Vendor",
    "guard": "Security Guard",
}

# permission -> roles that hold it
PERMISSIONS: Dict[str, set] = {
    "recommend":        set(ROLES),                       # parking recommendations, navigation
    "view_occupancy":   set(ROLES),                       # zone counts, forecasts, campus grounds
    "view_ops":         {"admin", "warden", "guard"},     # estate console (alerts, telemetry)
    "override_zone":    {"admin", "warden"},              # mark a zone closed / full
    "report_violation": {"admin", "warden", "guard"},     # file a parking violation report
    "view_reports":     {"admin", "warden", "guard"},
    "resolve_reports":  {"admin", "warden"},
    "camera_control":   {"admin"},                        # camera sources, ROI, detection, ingest
    "view_analytics":   {"admin"},                        # model benchmarks, SHAP, heatmaps
    "manage_calendar":  {"admin"},                        # enter events, exam windows and holidays
    "view_model":       {"admin"},                        # model version, live accuracy, data sources
    "manage_model":     {"admin"},                        # reload a newly trained model without restarting
    "manage_users":     {"admin"},
    "view_audit":       {"admin"},
}


def permissions_for(role: str) -> List[str]:
    return sorted(p for p, roles in PERMISSIONS.items() if role in roles)


def has_permission(user: Optional[Dict[str, Any]], perm: str) -> bool:
    return bool(user) and user["role"] in PERMISSIONS.get(perm, set())


# Where each role lands after logging in.
LANDING_PAGE = {"admin": "/admin", "warden": "/admin", "guard": "/admin"}


def landing_page(role: str) -> str:
    return LANDING_PAGE.get(role, "/")


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

COOKIE_NAME = "lpu_session"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def session_ttl() -> timedelta:
    return timedelta(hours=_env_int("SESSION_TTL_HOURS", 12))


def max_failed_logins() -> int:
    return _env_int("LOGIN_MAX_FAILURES", 5)


def lockout_duration() -> timedelta:
    return timedelta(minutes=_env_int("LOGIN_LOCKOUT_MINUTES", 15))


def cookie_secure() -> bool:
    return os.environ.get("COOKIE_SECURE", "0").lower() in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# Time helpers (all timestamps stored as ISO-8601 UTC)
# ---------------------------------------------------------------------------

def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds") if dt else None


def parse_iso(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    dt = datetime.fromisoformat(s)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Passwords
# ---------------------------------------------------------------------------

_SCRYPT_N, _SCRYPT_R, _SCRYPT_P = 2 ** 14, 8, 1
_COMMON_PASSWORDS = {
    "password", "password1", "password123", "12345678", "123456789", "1234567890",
    "qwerty123", "iloveyou", "admin123", "welcome1", "letmein1", "lpu12345", "parking1",
}


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, dklen=32)
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${_b64(salt)}${_b64(dk)}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, n, r, p, salt, dk = stored.split("$")
        if algo != "scrypt":
            return False
        expected = base64.b64decode(dk)
        calc = hashlib.scrypt(password.encode("utf-8"), salt=base64.b64decode(salt),
                              n=int(n), r=int(r), p=int(p), dklen=len(expected),
                              maxmem=64 * 1024 * 1024)
        return hmac.compare_digest(calc, expected)
    except Exception:
        return False


# Used to spend the same time on unknown usernames as on real ones.
_DUMMY_HASH = hash_password(secrets.token_urlsafe(16))


def min_password_length(role: str) -> int:
    return 12 if role == "admin" else 8


def validate_password(password: str, username: str, role: str) -> None:
    """Raise ValueError with a user-facing message if the password is too weak."""
    min_len = min_password_length(role)
    if len(password) < min_len:
        raise ValueError(f"Password must be at least {min_len} characters.")
    if len(password) > 128:
        raise ValueError("Password must be at most 128 characters.")
    if not re.search(r"[A-Za-z]", password) or not re.search(r"\d", password):
        raise ValueError("Password must contain at least one letter and one digit.")
    if password.lower() == username.lower() or username.lower() in password.lower():
        raise ValueError("Password must not contain your user ID.")
    if password.lower() in _COMMON_PASSWORDS:
        raise ValueError("That password is too common.")


_TEMP_ALPHABET = "abcdefghjkmnpqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789"


def generate_temp_password() -> str:
    """One-time password like `k7Qm-xT4p-9aZr` (no look-alike characters)."""
    while True:
        raw = "".join(secrets.choice(_TEMP_ALPHABET) for _ in range(12))
        if re.search(r"[A-Za-z]", raw) and re.search(r"\d", raw):
            return f"{raw[0:4]}-{raw[4:8]}-{raw[8:12]}"


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

USERNAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,39}$")

_PUBLIC_FIELDS = ("id", "username", "full_name", "role", "home_block", "department",
                  "must_change_password", "is_active", "expires_at", "created_at",
                  "updated_at", "last_login_at", "locked_until")


def public_user(row) -> Dict[str, Any]:
    u = {k: row[k] for k in _PUBLIC_FIELDS}
    u["must_change_password"] = bool(u["must_change_password"])
    u["is_active"] = bool(u["is_active"])
    u["role_label"] = ROLE_LABELS.get(u["role"], u["role"])
    exp = parse_iso(u["expires_at"])
    u["is_expired"] = bool(exp and exp <= now_utc())
    locked = parse_iso(u["locked_until"])
    u["is_locked"] = bool(locked and locked > now_utc())
    return u


def validate_username(username: str) -> str:
    username = (username or "").strip()
    if not USERNAME_RE.match(username):
        raise ValueError("User ID must be 3-40 characters: letters, digits, '.', '_' or '-'.")
    return username


def validate_role(role: str) -> str:
    role = (role or "").strip().lower()
    if role not in ROLES:
        raise ValueError(f"Role must be one of: {', '.join(ROLES)}.")
    return role


def get_user(user_id: int) -> Optional[Dict[str, Any]]:
    with get_db().conn() as c:
        row = c.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return public_user(row) if row else None


def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    with get_db().conn() as c:
        row = c.execute("SELECT * FROM users WHERE username = ?", (username.strip(),)).fetchone()
    return public_user(row) if row else None


def list_users(role: Optional[str] = None, q: Optional[str] = None) -> List[Dict[str, Any]]:
    sql, args = "SELECT * FROM users WHERE 1=1", []
    if role:
        sql += " AND role = ?"
        args.append(role)
    if q:
        sql += " AND (username LIKE ? OR full_name LIKE ? OR department LIKE ? OR home_block LIKE ?)"
        args += [f"%{q}%"] * 4
    sql += " ORDER BY role, username"
    with get_db().conn() as c:
        return [public_user(r) for r in c.execute(sql, args).fetchall()]


def count_users(role: Optional[str] = None, active_only: bool = False) -> int:
    sql, args = "SELECT COUNT(*) FROM users WHERE 1=1", []
    if role:
        sql += " AND role = ?"
        args.append(role)
    if active_only:
        sql += " AND is_active = 1"
    with get_db().conn() as c:
        return c.execute(sql, args).fetchone()[0]


def create_user(username: str, role: str, full_name: str = "", home_block: Optional[str] = None,
                department: Optional[str] = None, expires_at: Optional[str] = None,
                password: Optional[str] = None) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Create an account. Without `password`, a one-time password is generated, returned once and
    must be changed at first login. Raises ValueError on invalid input or a duplicate user ID.
    """
    username = validate_username(username)
    role = validate_role(role)
    temp = None
    if password is None:
        temp = password = generate_temp_password()
        must_change = 1
    else:
        validate_password(password, username, role)
        must_change = 0
    ts = iso(now_utc())
    try:
        with get_db().conn() as c:
            cur = c.execute(
                "INSERT INTO users (username, full_name, role, home_block, department, password_hash,"
                " must_change_password, expires_at, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (username, (full_name or "").strip(), role, home_block or None, (department or "").strip() or None,
                 hash_password(password), must_change, expires_at, ts, ts))
            uid = cur.lastrowid
    except Exception as e:
        if "UNIQUE" in str(e):
            raise ValueError(f"User ID '{username}' already exists.")
        raise
    return get_user(uid), temp


_UPDATABLE = {"full_name", "role", "home_block", "department", "is_active", "expires_at"}


def update_user(user_id: int, **fields) -> Optional[Dict[str, Any]]:
    changes = {k: v for k, v in fields.items() if k in _UPDATABLE}
    if "role" in changes:
        changes["role"] = validate_role(changes["role"])
    if "is_active" in changes:
        changes["is_active"] = 1 if changes["is_active"] else 0
    before = get_user(user_id)
    if not changes or before is None:
        return before
    changes["updated_at"] = iso(now_utc())
    cols = ", ".join(f"{k} = ?" for k in changes)
    with get_db().conn() as c:
        c.execute(f"UPDATE users SET {cols} WHERE id = ?", (*changes.values(), user_id))
    if changes.get("is_active") == 0 or changes.get("role", before["role"]) != before["role"]:
        revoke_sessions(user_id)   # a demoted or disabled user must log in again
    return get_user(user_id)


def set_password(user_id: int, password: str, must_change: bool) -> None:
    ts = iso(now_utc())
    with get_db().conn() as c:
        c.execute("UPDATE users SET password_hash = ?, must_change_password = ?, failed_attempts = 0,"
                  " locked_until = NULL, password_changed_at = ?, updated_at = ? WHERE id = ?",
                  (hash_password(password), 1 if must_change else 0, ts, ts, user_id))


def reset_password(user_id: int) -> str:
    temp = generate_temp_password()
    set_password(user_id, temp, must_change=True)
    revoke_sessions(user_id)
    return temp


def unlock_user(user_id: int) -> None:
    with get_db().conn() as c:
        c.execute("UPDATE users SET failed_attempts = 0, locked_until = NULL WHERE id = ?", (user_id,))


def delete_user(user_id: int) -> None:
    with get_db().conn() as c:
        c.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
        c.execute("DELETE FROM users WHERE id = ?", (user_id,))


def change_own_password(user_id: int, current: str, new: str) -> None:
    with get_db().conn() as c:
        row = c.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    if not row or not verify_password(current, row["password_hash"]):
        raise ValueError("Current password is incorrect.")
    if current == new:
        raise ValueError("New password must be different from the current one.")
    validate_password(new, row["username"], row["role"])
    set_password(user_id, new, must_change=False)


# ---------------------------------------------------------------------------
# Login, lockout, sessions
# ---------------------------------------------------------------------------

class LoginError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


GENERIC_LOGIN_ERROR = "Invalid user ID or password."


def authenticate(username: str, password: str) -> Dict[str, Any]:
    """Return the user on success; raise LoginError otherwise."""
    with get_db().conn() as c:
        row = c.execute("SELECT * FROM users WHERE username = ?", ((username or "").strip(),)).fetchone()
    if row is None:
        verify_password(password or "", _DUMMY_HASH)
        raise LoginError("invalid", GENERIC_LOGIN_ERROR)

    now = now_utc()
    locked = parse_iso(row["locked_until"])
    if locked and locked > now:
        mins = max(1, int((locked - now).total_seconds() // 60) + 1)
        raise LoginError("locked", f"Too many failed attempts. Try again in {mins} minute(s) or ask an admin to unlock.")

    if not verify_password(password or "", row["password_hash"]):
        failures = row["failed_attempts"] + 1
        lock_until = iso(now + lockout_duration()) if failures >= max_failed_logins() else None
        with get_db().conn() as c:
            c.execute("UPDATE users SET failed_attempts = ?, locked_until = ? WHERE id = ?",
                      (0 if lock_until else failures, lock_until, row["id"]))
        if lock_until:
            raise LoginError("locked", "Too many failed attempts. The account is locked for "
                                       f"{int(lockout_duration().total_seconds() // 60)} minutes.")
        raise LoginError("invalid", GENERIC_LOGIN_ERROR)

    # Correct password: only now reveal account state.
    if not row["is_active"]:
        raise LoginError("inactive", "This account has been disabled. Contact the parking administrator.")
    exp = parse_iso(row["expires_at"])
    if exp and exp <= now:
        raise LoginError("expired", "This account has expired. Contact the parking administrator.")

    with get_db().conn() as c:
        c.execute("UPDATE users SET failed_attempts = 0, locked_until = NULL, last_login_at = ? WHERE id = ?",
                  (iso(now), row["id"]))
    return get_user(row["id"])


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def create_session(user_id: int, user_agent: str = "") -> str:
    token = secrets.token_urlsafe(32)
    now = now_utc()
    with get_db().conn() as c:
        c.execute("DELETE FROM sessions WHERE expires_at <= ?", (iso(now),))
        c.execute("INSERT INTO sessions (token_hash, user_id, created_at, expires_at, user_agent) VALUES (?, ?, ?, ?, ?)",
                  (_token_hash(token), user_id, iso(now), iso(now + session_ttl()), (user_agent or "")[:200]))
    return token


def user_for_token(token: Optional[str]) -> Optional[Dict[str, Any]]:
    if not token:
        return None
    with get_db().conn() as c:
        row = c.execute(
            "SELECT u.*, s.expires_at AS session_expires FROM sessions s JOIN users u ON u.id = s.user_id"
            " WHERE s.token_hash = ?", (_token_hash(token),)).fetchone()
    if row is None:
        return None
    now = now_utc()
    acct_exp = parse_iso(row["expires_at"])
    if parse_iso(row["session_expires"]) <= now or not row["is_active"] or (acct_exp and acct_exp <= now):
        delete_session(token)
        return None
    return public_user(row)


def delete_session(token: str) -> None:
    with get_db().conn() as c:
        c.execute("DELETE FROM sessions WHERE token_hash = ?", (_token_hash(token),))


def revoke_sessions(user_id: int, keep_token: Optional[str] = None) -> None:
    with get_db().conn() as c:
        if keep_token:
            c.execute("DELETE FROM sessions WHERE user_id = ? AND token_hash != ?", (user_id, _token_hash(keep_token)))
        else:
            c.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

def client_ip(request: Optional[Request]) -> Optional[str]:
    if request is None or request.client is None:
        return None
    return request.client.host


def audit(action: str, actor: Optional[Dict[str, Any]] = None, target: Optional[str] = None,
          detail: Optional[Dict[str, Any]] = None, request: Optional[Request] = None,
          actor_username: Optional[str] = None) -> None:
    try:
        with get_db().conn() as c:
            c.execute("INSERT INTO audit_log (ts, actor_id, actor_username, action, target, detail, ip)"
                      " VALUES (?, ?, ?, ?, ?, ?, ?)",
                      (iso(now_utc()), actor["id"] if actor else None,
                       actor["username"] if actor else actor_username, action, target,
                       json.dumps(detail, default=str) if detail else None, client_ip(request)))
    except Exception as e:   # auditing must never break the request
        logger.error("Audit write failed for %s: %s", action, e)


def list_audit(limit: int = 100, offset: int = 0, action: Optional[str] = None,
               actor: Optional[str] = None) -> List[Dict[str, Any]]:
    sql, args = "SELECT * FROM audit_log WHERE 1=1", []
    if action:
        sql += " AND action = ?"
        args.append(action)
    if actor:
        sql += " AND actor_username LIKE ?"
        args.append(f"%{actor}%")
    sql += " ORDER BY id DESC LIMIT ? OFFSET ?"
    args += [limit, offset]
    with get_db().conn() as c:
        rows = c.execute(sql, args).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["detail"] = json.loads(d["detail"]) if d["detail"] else None
        out.append(d)
    return out


# ---------------------------------------------------------------------------
# First-run bootstrap
# ---------------------------------------------------------------------------

def initial_admin_file() -> str:
    db_path = get_db().path
    base = os.path.dirname(os.path.abspath(db_path)) if db_path != ":memory:" else os.getcwd()
    return os.path.join(base, "initial_admin_password.txt")


def ensure_bootstrap_admin() -> None:
    """
    With no accounts yet, create the first admin. Its password comes from ADMIN_PASSWORD, or is
    generated, logged once and written to initial_admin_password.txt next to the database (the
    file is removed when that admin sets their own password).
    """
    if count_users() > 0:
        return
    username = os.environ.get("ADMIN_USERNAME", "admin")
    env_pw = os.environ.get("ADMIN_PASSWORD")
    if env_pw:
        user, _ = create_user(username, "admin", full_name="Parking Administrator", password=env_pw)
        logger.warning("Created initial admin account '%s' from ADMIN_PASSWORD.", username)
    else:
        user, temp = create_user(username, "admin", full_name="Parking Administrator")
        path = initial_admin_file()
        if get_db().path != ":memory:":
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"user_id: {username}\none_time_password: {temp}\n"
                        "You must change this password at first login. This file is deleted afterwards.\n")
        logger.warning("Created initial admin '%s' with one-time password %s (also in %s). "
                       "Change it at first login.", username, temp, path)
    audit("bootstrap_admin_created", actor_username="system", target=username)


def clear_initial_admin_file() -> None:
    try:
        os.remove(initial_admin_file())
    except OSError:
        pass


# ---------------------------------------------------------------------------
# FastAPI dependencies
# ---------------------------------------------------------------------------

def token_from_request(request: Request) -> Optional[str]:
    token = request.cookies.get(COOKIE_NAME)
    if token:
        return token
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return None


def current_user(request: Request) -> Optional[Dict[str, Any]]:
    """The logged-in user, or None. Cached on the request."""
    if not hasattr(request.state, "user"):
        user = user_for_token(token_from_request(request))
        if user:
            user["permissions"] = permissions_for(user["role"])
        request.state.user = user
    return request.state.user


# Endpoints a user with a one-time password may still call.
_PW_CHANGE_ALLOWED = ("/api/auth/", "/api/blocks")   # blocks: read-only list the account page needs to render


def require_user(request: Request) -> Dict[str, Any]:
    user = current_user(request)
    if user is None:
        raise HTTPException(status_code=401, detail="Not logged in.")
    if user["must_change_password"] and not request.url.path.startswith(_PW_CHANGE_ALLOWED):
        raise HTTPException(status_code=403, detail="password_change_required")
    return user


def require(perm: str):
    if perm not in PERMISSIONS:
        raise ValueError(f"Unknown permission '{perm}'")

    def dependency(request: Request) -> Dict[str, Any]:
        user = require_user(request)
        if user["role"] not in PERMISSIONS[perm]:
            raise HTTPException(status_code=403, detail=f"Your role ({ROLE_LABELS[user['role']]}) cannot do this.")
        return user

    dependency.__name__ = f"require_{perm}"
    return dependency
