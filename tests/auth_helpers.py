"""Helpers for logging test clients in as a given role. Imports the app lazily (see conftest)."""

ADMIN_PASSWORD = "Root-Test-Pass-2026"
USER_PASSWORD = "Test-Pass-1234"


def ensure_admin():
    from api import auth
    auth.ensure_bootstrap_admin()


def make_user(username: str, role: str, password: str = USER_PASSWORD, **fields):
    """Create (or reset) an account with a known password that needs no change at login."""
    from api import auth
    ensure_admin()
    existing = auth.get_user_by_username(username)
    if existing is None:
        user, _ = auth.create_user(username, role, password=password, **fields)
    else:
        auth.update_user(existing["id"], role=role, is_active=True, **fields)
        auth.set_password(existing["id"], password, must_change=False)
        auth.unlock_user(existing["id"])
        user = auth.get_user(existing["id"])
    return user


def login_client(username: str = "admin", password: str = ADMIN_PASSWORD):
    from fastapi.testclient import TestClient
    from api.main import app
    ensure_admin()
    c = TestClient(app)
    r = c.post("/api/auth/login", json={"username": username, "password": password})
    assert r.status_code == 200, r.text
    return c


def client_for_role(role: str, username: str = None, **fields):
    username = username or f"t_{role}"
    make_user(username, role, **fields)
    return login_client(username, USER_PASSWORD)
