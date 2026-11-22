"""
Forecasting support endpoints: the campus calendar (events, exam windows, holidays) and model health.

The calendar is what lets the forecaster know ahead of time that an exam week, holiday or event is
coming; admins enter it here and predictions change on the next request.
"""
import os
from datetime import date, datetime, time, timedelta
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from api import auth, calendar_store, model_monitor
from api.routes import ZONES_MAP, get_store
from features import calendar as cal
from models import inference, live
from recommendation.campus_blocks import get_campus_blocks

_IST = ZoneInfo("Asia/Kolkata")
router = APIRouter(prefix="/api")
MAX_EVENT_HOURS = 72
MAX_RANGE_DAYS = 60
MAX_VEHICLES = 5000


def _bad(msg: str, status: int = 422):
    raise HTTPException(status_code=status, detail=msg)


class CalendarCreate(BaseModel):
    kind: str = Field(..., description="exam | holiday | event")
    name: str = Field(..., min_length=2, max_length=80)
    start: str = Field(..., description="YYYY-MM-DD (exam/holiday) or YYYY-MM-DDTHH:MM (event)")
    end: Optional[str] = Field(None, description="Same format; defaults to the start day (exam/holiday)")
    venue_block: Optional[str] = Field(None, description="Events: the block the event is at")
    expected_vehicles: Optional[int] = Field(None, ge=1, le=MAX_VEHICLES, description="Events: expected extra vehicles")
    notes: str = Field("", max_length=300)


def _parse(value: str, field: str) -> datetime:
    try:
        v = value.strip()
        return datetime.fromisoformat(v if len(v) > 10 else v + "T00:00:00")
    except ValueError:
        _bad(f"{field} must be YYYY-MM-DD or YYYY-MM-DDTHH:MM.")


def _build(req: CalendarCreate) -> cal.CalendarEntry:
    kind = req.kind.strip().lower()
    if kind not in cal.KINDS:
        _bad(f"kind must be one of {list(cal.KINDS)}.")
    start = _parse(req.start, "start")
    if start.year < 2020 or start.year > 2100:
        _bad("start is out of range.")
    if kind in ("exam", "holiday"):
        end = _parse(req.end, "end") if req.end else start
        if end.date() < start.date():
            _bad("end must not be before start.")
        if (end.date() - start.date()).days > MAX_RANGE_DAYS:
            _bad(f"A {kind} window can be at most {MAX_RANGE_DAYS} days long.")
        return cal.day_range(start.date(), end.date(), kind, req.name.strip())
    # event
    if len(req.start.strip()) <= 10:
        _bad("Events need a start time, e.g. 2026-12-01T09:00.")
    if not req.end or len(req.end.strip()) <= 10:
        _bad("Events need an end time, e.g. 2026-12-01T17:00.")
    end = _parse(req.end, "end")
    if end <= start:
        _bad("The event must end after it starts.")
    if end - start > timedelta(hours=MAX_EVENT_HOURS):
        _bad(f"An event can last at most {MAX_EVENT_HOURS} hours; enter longer ones as several events.")
    if not req.venue_block:
        _bad("Events need a venue block.")
    block = get_campus_blocks().canonical_id(req.venue_block)
    if block is None:
        _bad(f"Unknown venue block '{req.venue_block}'. See GET /api/blocks.")
    if not req.expected_vehicles:
        _bad("Events need an estimate of the extra vehicles expected.")
    return cal.CalendarEntry("event", req.name.strip(), start, end, venue_block=block,
                             expected_vehicles=int(req.expected_vehicles))


@router.get("/calendar")
def list_calendar(days_back: int = Query(7, ge=0, le=365), days_ahead: int = Query(60, ge=1, le=365),
                  user=Depends(auth.require("view_occupancy"))):
    now = datetime.now(_IST).replace(tzinfo=None)
    entries = calendar_store.entries_between(now - timedelta(days=days_back), now + timedelta(days=days_ahead))
    return {"entries": [calendar_store.to_dict(e) for e in entries], "context_now": cal.context_at(now)}


@router.get("/calendar/context")
def calendar_context(user=Depends(auth.require("view_occupancy"))):
    """What is special about right now (bell window, exam day, holiday, events on or starting soon)."""
    now = datetime.now(_IST).replace(tzinfo=None)
    return {"now": now.isoformat(timespec="minutes"), **cal.context_at(now, calendar_store.list_entries())}


@router.post("/calendar/preview")
def preview_calendar_entry(req: CalendarCreate, user=Depends(auth.require("manage_calendar"))):
    """Which grounds an event would load, without saving it."""
    entry = _build(req)
    return {"entry": calendar_store.to_dict(entry)}


@router.post("/calendar", status_code=201)
def create_calendar_entry(req: CalendarCreate, request: Request, user=Depends(auth.require("manage_calendar"))):
    entry = _build(req)
    try:
        saved = calendar_store.add_entry(entry.kind, entry.name, entry.start, entry.end, entry.venue_block,
                                         entry.expected_vehicles, req.notes, user["username"])
    except ValueError as e:
        _bad(str(e))
    live.bump()
    auth.audit("calendar_entry_created", actor=user, target=f"calendar:{saved.id}",
               detail={"kind": saved.kind, "name": saved.name, "start": saved.start.isoformat(),
                       "end": saved.end.isoformat(), "venue": saved.venue_block, "vehicles": saved.expected_vehicles},
               request=request)
    return calendar_store.to_dict(saved)


@router.delete("/calendar/{entry_id}")
def delete_calendar_entry(entry_id: int, request: Request, user=Depends(auth.require("manage_calendar"))):
    existing = calendar_store.get_entry(entry_id)
    if existing is None:
        _bad("Calendar entry not found.", 404)
    calendar_store.delete_entry(entry_id)
    live.bump()
    auth.audit("calendar_entry_deleted", actor=user, target=f"calendar:{entry_id}",
               detail={"kind": existing.kind, "name": existing.name}, request=request)
    return {"status": "deleted", "id": entry_id}


# ---------------------------------------------------------------------------
# Model health
# ---------------------------------------------------------------------------

def _csv_rows(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    return df.where(df.notna(), None).to_dict(orient="records")


@router.post("/model/reload")
def reload_model(request: Request, user=Depends(auth.require("manage_model"))):
    """Load the artifacts on disk (e.g. after `python -m models.retrain` promoted a new model) without a restart."""
    before = inference.model_meta().get("version")
    ok = inference._load_models(force=True)
    if not ok:
        _bad(f"Could not load the model artifacts: {inference.models_ready()['error']}", 503)
    live.bump()
    after = inference.model_meta().get("version")
    auth.audit("model_reloaded", actor=user, target=str(after), detail={"from": before, "to": after}, request=request)
    return {"status": "reloaded", "previous_version": before, "version": after, "changed": before != after}


@router.get("/model/status")
def model_status(days: int = Query(7, ge=1, le=60), user=Depends(auth.require("view_model"))):
    """Model version and training data, where each ground's count comes from, live accuracy and drift."""
    ready = inference.models_ready()
    meta = inference.model_meta()
    zones = live.predict_all(ZONES_MAP, None, 15)
    return {
        "loaded": ready["loaded"], "error": ready["error"],
        "meta": {k: meta.get(k) for k in ("version", "trained_at", "data_source", "data_from", "data_to", "rows",
                                           "zones", "features", "intervals", "libraries", "notes")},
        "backtest": meta.get("metrics_test", {}),
        "regime_metrics": _csv_rows("artifacts/regime_metrics.csv"),
        "data_mode": live.data_mode(ZONES_MAP),
        "zones": [{"zone_id": z, "name": ZONES_MAP[z]["name"], "count_source": p["count_source"],
                   "camera_status": p["camera_status"], "age_seconds": p["camera_age_seconds"],
                   "current_count": p["current_count"], "confidence": p["confidence"], "frozen": p["stream_frozen"]}
                  for z, p in zones.items()],
        "live_accuracy": model_monitor.live_accuracy(get_store(), days),
    }
