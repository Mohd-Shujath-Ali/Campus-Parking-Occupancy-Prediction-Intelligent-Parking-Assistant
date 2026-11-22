"""
Database-backed campus calendar (exam windows, holidays, events) for serving.

The forecaster reads it through features.calendar.set_provider(), so a new event entered by an
admin changes predictions on the next request. Entries are cached for a few seconds and the cache
is dropped on every write.
"""
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from api import auth
from api.db import get_db
from features.calendar import CalendarEntry, KINDS, event_impact

_CACHE_TTL_S = 10.0
_cache: Dict[str, Any] = {"at": 0.0, "entries": []}
_version = 0                       # bumped on every write; the live predictor keys its cache on it


def version() -> int:
    return _version


def _row_to_entry(r) -> CalendarEntry:
    return CalendarEntry(kind=r["kind"], name=r["name"], start=datetime.fromisoformat(r["start_ts"]),
                         end=datetime.fromisoformat(r["end_ts"]), venue_block=r["venue_block"],
                         expected_vehicles=int(r["expected_vehicles"]), id=r["id"])


def list_entries() -> List[CalendarEntry]:
    """All entries (cached briefly). Used as the forecaster's calendar provider."""
    now = time.monotonic()
    if now - _cache["at"] > _CACHE_TTL_S:
        with get_db().conn() as c:
            rows = c.execute("SELECT * FROM calendar_entries ORDER BY start_ts").fetchall()
        _cache["entries"] = [_row_to_entry(r) for r in rows]
        _cache["at"] = now
    return list(_cache["entries"])


def _invalidate():
    global _version
    _cache["at"] = 0.0
    _version += 1


def to_dict(row_or_entry) -> Dict[str, Any]:
    e = row_or_entry if isinstance(row_or_entry, CalendarEntry) else _row_to_entry(row_or_entry)
    d = {"id": e.id, "kind": e.kind, "name": e.name, "start": e.start.isoformat(timespec="minutes"),
         "end": e.end.isoformat(timespec="minutes"), "venue_block": e.venue_block,
         "expected_vehicles": e.expected_vehicles}
    if e.kind == "event" and e.venue_block:
        d["impact"] = event_impact(e)
    return d


def add_entry(kind: str, name: str, start: datetime, end: datetime, venue_block: Optional[str] = None,
              expected_vehicles: int = 0, notes: str = "", created_by: Optional[str] = None) -> CalendarEntry:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}.")
    if end < start:
        raise ValueError("End must not be before start.")
    with get_db().conn() as c:
        cur = c.execute(
            "INSERT INTO calendar_entries (kind, name, start_ts, end_ts, venue_block, expected_vehicles, notes,"
            " created_by, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (kind, name.strip(), start.isoformat(timespec="seconds"), end.isoformat(timespec="seconds"),
             venue_block, int(expected_vehicles), notes.strip() or None, created_by, auth.iso(auth.now_utc())))
        row = c.execute("SELECT * FROM calendar_entries WHERE id = ?", (cur.lastrowid,)).fetchone()
    _invalidate()
    return _row_to_entry(row)


def get_entry(entry_id: int) -> Optional[CalendarEntry]:
    with get_db().conn() as c:
        row = c.execute("SELECT * FROM calendar_entries WHERE id = ?", (entry_id,)).fetchone()
    return _row_to_entry(row) if row else None


def delete_entry(entry_id: int) -> bool:
    with get_db().conn() as c:
        n = c.execute("DELETE FROM calendar_entries WHERE id = ?", (entry_id,)).rowcount
    _invalidate()
    return bool(n)


def entries_between(start: datetime, end: datetime) -> List[CalendarEntry]:
    return [e for e in list_entries() if e.end >= start and e.start <= end]
