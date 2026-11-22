"""
Application database (SQLite): user accounts, sessions, audit log and operational state
(zone overrides, violation reports, camera configuration, versioned ROI polygons).

The path comes from the AUTH_DB environment variable (default data/live/app.db, which is
git-ignored). ":memory:" is supported for tests.
"""

import os
import sqlite3
import threading
from contextlib import contextmanager
from typing import Iterator, Optional

DEFAULT_APP_DB = "data/live/app.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    username             TEXT NOT NULL UNIQUE COLLATE NOCASE,
    full_name            TEXT NOT NULL DEFAULT '',
    role                 TEXT NOT NULL,
    home_block           TEXT,
    department           TEXT,
    password_hash        TEXT NOT NULL,
    must_change_password INTEGER NOT NULL DEFAULT 1,
    is_active            INTEGER NOT NULL DEFAULT 1,
    expires_at           TEXT,
    failed_attempts      INTEGER NOT NULL DEFAULT 0,
    locked_until         TEXT,
    created_at           TEXT NOT NULL,
    updated_at           TEXT NOT NULL,
    last_login_at        TEXT,
    password_changed_at  TEXT
);

CREATE TABLE IF NOT EXISTS sessions (
    token_hash  TEXT PRIMARY KEY,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at  TEXT NOT NULL,
    expires_at  TEXT NOT NULL,
    user_agent  TEXT
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);

CREATE TABLE IF NOT EXISTS audit_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT NOT NULL,
    actor_id        INTEGER,
    actor_username  TEXT,
    action          TEXT NOT NULL,
    target          TEXT,
    detail          TEXT,
    ip              TEXT
);
CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_log(ts);

CREATE TABLE IF NOT EXISTS zone_overrides (
    zone_id     TEXT PRIMARY KEY,
    status      TEXT NOT NULL,
    reason      TEXT,
    set_by      TEXT,
    set_at      TEXT NOT NULL,
    expires_at  TEXT
);

CREATE TABLE IF NOT EXISTS reports (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at         TEXT NOT NULL,
    reporter_id        INTEGER,
    reporter_username  TEXT,
    zone_id            TEXT NOT NULL,
    category           TEXT NOT NULL,
    vehicle_number     TEXT,
    description        TEXT,
    status             TEXT NOT NULL DEFAULT 'open',
    resolved_by        TEXT,
    resolved_at        TEXT,
    resolution_note    TEXT
);

CREATE TABLE IF NOT EXISTS cameras (
    camera_id    TEXT PRIMARY KEY,
    zone_id      TEXT NOT NULL,
    label        TEXT,
    source_type  TEXT NOT NULL DEFAULT 'none',
    source_url   TEXT,
    enabled      INTEGER NOT NULL DEFAULT 1,
    updated_by   TEXT,
    updated_at   TEXT
);

CREATE TABLE IF NOT EXISTS calendar_entries (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    kind               TEXT NOT NULL,              -- exam | holiday | event
    name               TEXT NOT NULL,
    start_ts           TEXT NOT NULL,              -- naive campus-local (IST) ISO time
    end_ts             TEXT NOT NULL,
    venue_block        TEXT,
    expected_vehicles  INTEGER NOT NULL DEFAULT 0,
    notes              TEXT,
    created_by         TEXT,
    created_at         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_calendar_range ON calendar_entries(start_ts, end_ts);

CREATE TABLE IF NOT EXISTS prediction_log (
    zone_id        TEXT NOT NULL,
    made_bucket    TEXT NOT NULL,                  -- 5-minute bucket the forecast was made in (naive IST)
    horizon_min    INTEGER NOT NULL,
    target_bucket  TEXT NOT NULL,
    predicted      REAL NOT NULL,
    low            REAL,
    high           REAL,
    p_free         REAL,
    capacity       INTEGER NOT NULL,
    model_version  TEXT,
    PRIMARY KEY (zone_id, made_bucket, horizon_min)
);
CREATE INDEX IF NOT EXISTS idx_predlog_target ON prediction_log(target_bucket);

CREATE TABLE IF NOT EXISTS roi_versions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id     TEXT NOT NULL,
    version       INTEGER NOT NULL,
    polygon       TEXT NOT NULL,
    image_width   INTEGER,
    image_height  INTEGER,
    note          TEXT,
    created_by    TEXT,
    created_at    TEXT NOT NULL,
    is_active     INTEGER NOT NULL DEFAULT 0,
    UNIQUE (camera_id, version)
);
"""


class AppDB:
    def __init__(self, path: str = DEFAULT_APP_DB):
        self.path = path
        self._lock = threading.Lock()
        self._mem: Optional[sqlite3.Connection] = None
        if path == ":memory:":
            self._mem = sqlite3.connect(":memory:", check_same_thread=False)
            self._mem.row_factory = sqlite3.Row
            self._mem.execute("PRAGMA foreign_keys=ON")
        else:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with self.conn() as c:
            if self._mem is None:
                c.execute("PRAGMA journal_mode=WAL")
            c.executescript(SCHEMA)

    @contextmanager
    def conn(self) -> Iterator[sqlite3.Connection]:
        if self._mem is not None:
            with self._lock:
                try:
                    yield self._mem
                    self._mem.commit()
                except Exception:
                    self._mem.rollback()
                    raise
            return
        c = sqlite3.connect(self.path, timeout=10)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA foreign_keys=ON")
        try:
            yield c
            c.commit()
        except Exception:
            c.rollback()
            raise
        finally:
            c.close()


_db: Optional[AppDB] = None
_db_lock = threading.Lock()


def get_db() -> AppDB:
    """Process-wide database handle, created on first use (so tests can set AUTH_DB first)."""
    global _db
    if _db is None:
        with _db_lock:
            if _db is None:
                _db = AppDB(os.environ.get("AUTH_DB", DEFAULT_APP_DB))
    return _db


def reset_db(path: Optional[str] = None) -> AppDB:
    """Replace the process-wide handle (tests only)."""
    global _db
    with _db_lock:
        _db = AppDB(path or os.environ.get("AUTH_DB", DEFAULT_APP_DB))
    return _db
