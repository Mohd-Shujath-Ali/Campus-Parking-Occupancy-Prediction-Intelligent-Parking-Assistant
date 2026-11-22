"""
Operational endpoints:

- Zone overrides (admin, warden): mark a ground CLOSED or FULL, optionally for N minutes.
  Recommendations push overridden grounds to the bottom and never suggest them.
- Violation reports (filed by admin / warden / guard, resolved by admin / warden).
- Cameras and ROI (admin only): camera source per zone camera, and versioned ROI polygons
  with one active version per camera and roll-back to any earlier version.
"""

import json
import re
from datetime import timedelta
from typing import Any, Dict, List, Optional

import yaml
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from api import auth
from api.db import get_db

router = APIRouter(prefix="/api")

with open("configs/zones.yaml", "r") as f:
    _ZONES = {z["id"]: z for z in yaml.safe_load(f)["zones"]}


def _bad(msg: str, status: int = 422):
    raise HTTPException(status_code=status, detail=msg)


def _zone_or_404(zone_id: str) -> Dict[str, Any]:
    z = _ZONES.get(zone_id)
    if z is None:
        _bad(f"Zone '{zone_id}' not found.", 404)
    return z


# ---------------------------------------------------------------------------
# Zone overrides
# ---------------------------------------------------------------------------

OVERRIDE_STATUSES = ("CLOSED", "FULL")


class OverrideRequest(BaseModel):
    status: str = Field(..., description="CLOSED (ground unusable) or FULL (no free space)")
    reason: str = Field("", max_length=200)
    minutes: Optional[int] = Field(None, ge=5, le=24 * 60, description="Auto-expire after N minutes; omit to keep until cleared")


def active_overrides() -> Dict[str, Dict[str, Any]]:
    """zone_id -> override, ignoring (and purging) expired ones."""
    now = auth.iso(auth.now_utc())
    with get_db().conn() as c:
        c.execute("DELETE FROM zone_overrides WHERE expires_at IS NOT NULL AND expires_at <= ?", (now,))
        rows = c.execute("SELECT * FROM zone_overrides").fetchall()
    return {r["zone_id"]: dict(r) for r in rows}


@router.get("/overrides")
def list_overrides(user=Depends(auth.require("view_occupancy"))):
    out = []
    for zid, o in active_overrides().items():
        out.append({**o, "zone_name": _ZONES.get(zid, {}).get("name", zid)})
    return {"overrides": out}


@router.put("/zones/{zone_id}/override")
def set_override(zone_id: str, req: OverrideRequest, request: Request, user=Depends(auth.require("override_zone"))):
    _zone_or_404(zone_id)
    status = req.status.strip().upper()
    if status not in OVERRIDE_STATUSES:
        _bad(f"status must be one of {OVERRIDE_STATUSES}.")
    now = auth.now_utc()
    expires = auth.iso(now + timedelta(minutes=req.minutes)) if req.minutes else None
    with get_db().conn() as c:
        c.execute("INSERT INTO zone_overrides (zone_id, status, reason, set_by, set_at, expires_at)"
                  " VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(zone_id) DO UPDATE SET status=excluded.status,"
                  " reason=excluded.reason, set_by=excluded.set_by, set_at=excluded.set_at, expires_at=excluded.expires_at",
                  (zone_id, status, req.reason.strip(), user["username"], auth.iso(now), expires))
    auth.audit("zone_override_set", actor=user, target=zone_id,
               detail={"status": status, "reason": req.reason, "minutes": req.minutes}, request=request)
    return active_overrides()[zone_id]


@router.delete("/zones/{zone_id}/override")
def clear_override(zone_id: str, request: Request, user=Depends(auth.require("override_zone"))):
    _zone_or_404(zone_id)
    with get_db().conn() as c:
        n = c.execute("DELETE FROM zone_overrides WHERE zone_id = ?", (zone_id,)).rowcount
    if n:
        auth.audit("zone_override_cleared", actor=user, target=zone_id, request=request)
    return {"status": "cleared" if n else "no_override", "zone_id": zone_id}


# ---------------------------------------------------------------------------
# Violation reports
# ---------------------------------------------------------------------------

REPORT_CATEGORIES = {
    "wrong_parking": "Parked outside a marked bay",
    "no_parking_zone": "Parked in a no-parking area",
    "blocking": "Blocking a lane, gate or another vehicle",
    "unauthorized_vehicle": "Vehicle without a campus pass",
    "overstay": "Overstayed the allowed time",
    "other": "Other",
}
_PLATE_RE = re.compile(r"^[A-Z0-9][A-Z0-9 -]{1,19}$")


class ReportCreate(BaseModel):
    zone_id: str
    category: str
    vehicle_number: Optional[str] = Field(None, max_length=20)
    description: str = Field("", max_length=500)


class ReportResolve(BaseModel):
    note: str = Field("", max_length=300)


@router.get("/reports/categories")
def report_categories(user=Depends(auth.require("report_violation"))):
    return [{"id": k, "label": v} for k, v in REPORT_CATEGORIES.items()]


@router.post("/reports", status_code=201)
def create_report(req: ReportCreate, request: Request, user=Depends(auth.require("report_violation"))):
    _zone_or_404(req.zone_id)
    if req.category not in REPORT_CATEGORIES:
        _bad(f"category must be one of {list(REPORT_CATEGORIES)}.")
    plate = (req.vehicle_number or "").strip().upper() or None
    if plate and not _PLATE_RE.match(plate):
        _bad("vehicle_number may only contain letters, digits, spaces and '-'.")
    with get_db().conn() as c:
        cur = c.execute("INSERT INTO reports (created_at, reporter_id, reporter_username, zone_id, category,"
                        " vehicle_number, description) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (auth.iso(auth.now_utc()), user["id"], user["username"], req.zone_id, req.category,
                         plate, req.description.strip()))
        rid = cur.lastrowid
        row = c.execute("SELECT * FROM reports WHERE id = ?", (rid,)).fetchone()
    auth.audit("report_created", actor=user, target=f"report:{rid}",
               detail={"zone_id": req.zone_id, "category": req.category}, request=request)
    return _report_out(row)


def _report_out(row) -> Dict[str, Any]:
    d = dict(row)
    d["zone_name"] = _ZONES.get(d["zone_id"], {}).get("name", d["zone_id"])
    d["category_label"] = REPORT_CATEGORIES.get(d["category"], d["category"])
    return d


@router.get("/reports")
def list_reports(status: Optional[str] = Query(None, description="open | resolved"),
                 limit: int = Query(100, ge=1, le=500), user=Depends(auth.require("view_reports"))):
    sql, args = "SELECT * FROM reports WHERE 1=1", []
    if status:
        if status not in ("open", "resolved"):
            _bad("status must be 'open' or 'resolved'.")
        sql += " AND status = ?"
        args.append(status)
    sql += " ORDER BY id DESC LIMIT ?"
    args.append(limit)
    with get_db().conn() as c:
        rows = c.execute(sql, args).fetchall()
        open_count = c.execute("SELECT COUNT(*) FROM reports WHERE status = 'open'").fetchone()[0]
    return {"reports": [_report_out(r) for r in rows], "open_count": open_count}


@router.post("/reports/{report_id}/resolve")
def resolve_report(report_id: int, req: ReportResolve, request: Request, user=Depends(auth.require("resolve_reports"))):
    with get_db().conn() as c:
        row = c.execute("SELECT * FROM reports WHERE id = ?", (report_id,)).fetchone()
        if row is None:
            _bad("Report not found.", 404)
        if row["status"] == "resolved":
            _bad("Report is already resolved.", 409)
        c.execute("UPDATE reports SET status = 'resolved', resolved_by = ?, resolved_at = ?, resolution_note = ?"
                  " WHERE id = ?", (user["username"], auth.iso(auth.now_utc()), req.note.strip(), report_id))
        row = c.execute("SELECT * FROM reports WHERE id = ?", (report_id,)).fetchone()
    auth.audit("report_resolved", actor=user, target=f"report:{report_id}", request=request)
    return _report_out(row)


# ---------------------------------------------------------------------------
# Cameras and ROI (admin only)
# ---------------------------------------------------------------------------

SOURCE_TYPES = ("none", "webcam", "rtsp", "http", "file")


class CameraUpdate(BaseModel):
    source_type: Optional[str] = None
    source_url: Optional[str] = Field(None, max_length=500)
    enabled: Optional[bool] = None
    label: Optional[str] = Field(None, max_length=80)


class RoiSave(BaseModel):
    polygon: List[List[float]] = Field(..., description="[[x, y], ...] in image pixel coordinates, at least 3 points")
    image_width: Optional[int] = Field(None, ge=1, le=10000)
    image_height: Optional[int] = Field(None, ge=1, le=10000)
    note: str = Field("", max_length=200)


def _seed_cameras() -> None:
    with get_db().conn() as c:
        for z in _ZONES.values():
            c.execute("INSERT OR IGNORE INTO cameras (camera_id, zone_id, label) VALUES (?, ?, ?)",
                      (z["camera_id"], z["id"], z["name"]))


def _camera_row(camera_id: str):
    _seed_cameras()
    with get_db().conn() as c:
        row = c.execute("SELECT * FROM cameras WHERE camera_id = ?", (camera_id,)).fetchone()
    if row is None:
        _bad(f"Camera '{camera_id}' not found.", 404)
    return row


def _roi_out(row) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    d = dict(row)
    d["polygon"] = json.loads(d["polygon"])
    d["is_active"] = bool(d["is_active"])
    return d


def active_roi(camera_id: str) -> Optional[Dict[str, Any]]:
    with get_db().conn() as c:
        row = c.execute("SELECT * FROM roi_versions WHERE camera_id = ? AND is_active = 1", (camera_id,)).fetchone()
    return _roi_out(row)


def _camera_out(row) -> Dict[str, Any]:
    d = dict(row)
    d["enabled"] = bool(d["enabled"])
    d["zone_name"] = _ZONES.get(d["zone_id"], {}).get("name", d["zone_id"])
    roi = active_roi(d["camera_id"])
    d["active_roi_version"] = roi["version"] if roi else None
    d["active_roi_points"] = len(roi["polygon"]) if roi else 0
    return d


@router.get("/cameras")
def list_cameras(user=Depends(auth.require("camera_control"))):
    _seed_cameras()
    with get_db().conn() as c:
        rows = c.execute("SELECT * FROM cameras ORDER BY zone_id, camera_id").fetchall()
    return {"cameras": [_camera_out(r) for r in rows], "source_types": list(SOURCE_TYPES)}


@router.patch("/cameras/{camera_id}")
def update_camera(camera_id: str, req: CameraUpdate, request: Request, user=Depends(auth.require("camera_control"))):
    row = _camera_row(camera_id)
    fields = req.model_dump(exclude_unset=True)
    src_type = fields.get("source_type", row["source_type"])
    if src_type not in SOURCE_TYPES:
        _bad(f"source_type must be one of {SOURCE_TYPES}.")
    url = fields.get("source_url", row["source_url"]) or ""
    url = url.strip()
    if src_type == "rtsp" and not url.lower().startswith(("rtsp://", "rtsps://")):
        _bad("An RTSP source needs a URL starting with rtsp:// or rtsps://.")
    if src_type == "http" and not url.lower().startswith(("http://", "https://")):
        _bad("An HTTP source needs a URL starting with http:// or https://.")
    if src_type in ("none", "webcam"):
        url = ""
    if "source_url" in fields or "source_type" in fields:
        fields["source_url"] = url or None
    if "enabled" in fields:
        fields["enabled"] = 1 if fields["enabled"] else 0
    if fields:
        fields.update(updated_by=user["username"], updated_at=auth.iso(auth.now_utc()))
        cols = ", ".join(f"{k} = ?" for k in fields)
        with get_db().conn() as c:
            c.execute(f"UPDATE cameras SET {cols} WHERE camera_id = ?", (*fields.values(), camera_id))
    # never write a stream URL (may embed credentials) into the audit log
    auth.audit("camera_updated", actor=user, target=camera_id,
               detail={k: v for k, v in fields.items() if k not in ("source_url", "updated_by", "updated_at")},
               request=request)
    return _camera_out(_camera_row(camera_id))


@router.get("/cameras/{camera_id}/roi")
def get_camera_roi(camera_id: str, user=Depends(auth.require("camera_control"))):
    _camera_row(camera_id)
    return {"camera_id": camera_id, "active": active_roi(camera_id)}


@router.get("/cameras/{camera_id}/roi/history")
def get_roi_history(camera_id: str, user=Depends(auth.require("camera_control"))):
    _camera_row(camera_id)
    with get_db().conn() as c:
        rows = c.execute("SELECT * FROM roi_versions WHERE camera_id = ? ORDER BY version DESC", (camera_id,)).fetchall()
    return {"camera_id": camera_id, "versions": [_roi_out(r) for r in rows]}


@router.put("/cameras/{camera_id}/roi", status_code=201)
def save_camera_roi(camera_id: str, req: RoiSave, request: Request, user=Depends(auth.require("camera_control"))):
    _camera_row(camera_id)
    pts = req.polygon
    if not (3 <= len(pts) <= 100):
        _bad("ROI polygon needs between 3 and 100 points.")
    for p in pts:
        if len(p) != 2 or p[0] < 0 or p[1] < 0:
            _bad("Each ROI point must be [x, y] with non-negative coordinates.")
        if (req.image_width and p[0] > req.image_width) or (req.image_height and p[1] > req.image_height):
            _bad("ROI point lies outside the image bounds.")
    polygon = [[round(float(x), 1), round(float(y), 1)] for x, y in pts]
    with get_db().conn() as c:
        version = (c.execute("SELECT COALESCE(MAX(version), 0) FROM roi_versions WHERE camera_id = ?",
                             (camera_id,)).fetchone()[0]) + 1
        c.execute("UPDATE roi_versions SET is_active = 0 WHERE camera_id = ?", (camera_id,))
        c.execute("INSERT INTO roi_versions (camera_id, version, polygon, image_width, image_height, note,"
                  " created_by, created_at, is_active) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)",
                  (camera_id, version, json.dumps(polygon), req.image_width, req.image_height,
                   req.note.strip(), user["username"], auth.iso(auth.now_utc())))
    auth.audit("roi_saved", actor=user, target=camera_id, detail={"version": version, "points": len(polygon)},
               request=request)
    return active_roi(camera_id)


@router.post("/cameras/{camera_id}/roi/{version}/activate")
def activate_roi(camera_id: str, version: int, request: Request, user=Depends(auth.require("camera_control"))):
    _camera_row(camera_id)
    with get_db().conn() as c:
        exists = c.execute("SELECT 1 FROM roi_versions WHERE camera_id = ? AND version = ?",
                           (camera_id, version)).fetchone()
        if not exists:
            _bad(f"ROI version {version} not found for {camera_id}.", 404)
        c.execute("UPDATE roi_versions SET is_active = (version = ?) WHERE camera_id = ?", (version, camera_id))
    auth.audit("roi_activated", actor=user, target=camera_id, detail={"version": version}, request=request)
    return active_roi(camera_id)
