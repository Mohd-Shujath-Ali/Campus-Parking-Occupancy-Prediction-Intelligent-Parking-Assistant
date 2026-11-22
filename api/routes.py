"""
LPU Campus Parking Intelligence System — FastAPI Routes
All endpoints use the REAL XGBoost model for predictions.
No hardcoded heuristics remain in production paths.
"""

import os
import hmac
import json
import yaml
import logging
import pandas as pd
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from fastapi import APIRouter, HTTPException, Query, UploadFile, File, Form, Request, Depends
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, FileResponse
from typing import List, Dict, Any, Optional

from fastapi import Header
from api import auth, calendar_store
from api.auth import require
from api.limiter import limiter
from api.ops_routes import active_overrides
from features import calendar as cal
from recommendation.campus_blocks import get_campus_blocks
from utils.config import load_model_config
from api.schemas import (PredictionRequest, PredictionResponse, RecommendationResponse, ZoneRankItem,
                         IngestRequest, NavigateRequest)
from data_collection.store import ObservationStore, DEFAULT_DB
from recommendation.parking_ranker import ParkingRanker
from models.inference import predict_zone, models_ready, model_meta, _build_feature_row
from models import live
from models.history import HISTORY
from models import analysis as zone_analytics
from models.demand import summarise_days

logger = logging.getLogger(__name__)
_IST = ZoneInfo("Asia/Kolkata")

router = APIRouter(prefix="/api")

# ---------------------------------------------------------------------------
# Load campus topology
# ---------------------------------------------------------------------------
with open("configs/zones.yaml", "r") as f:
    ZONES_CONFIG = yaml.safe_load(f)

ZONES_MAP = {z["id"]: z for z in ZONES_CONFIG["zones"]}
ranker = ParkingRanker()


# ---------------------------------------------------------------------------
# Health & Status
# ---------------------------------------------------------------------------

_NO_SERVER_VISION = ("Server-side detection (OpenCV/Ultralytics) is not installed in this deployment. "
                     "The Vision page runs YOLO in your browser instead.")


def _server_vision():
    """The server-side detection module, or None when OpenCV/Ultralytics are not installed
    (e.g. the slim serverless deployment, where YOLO runs in the browser)."""
    try:
        from computer_vision import detection
        return detection
    except ImportError:
        return None


@router.get("/health")
def get_health():
    ml_status = models_ready()
    vision = _server_vision()
    cv_status = vision.model_status() if vision else {"loaded": False, "server_side": False,
                                                      "note": _NO_SERVER_VISION}
    return {
        "status": "healthy",
        "service": "LPU Campus Parking Intelligence System",
        "campus": "Lovely Professional University (Phagwara, Punjab)",
        "total_monitored_zones": len(ZONES_MAP),
        "forecast_horizons_minutes": [15, 30, 60],
        "ml_model": {
            "xgboost_loaded": ml_status["loaded"],
            "error": ml_status["error"],
            "version": model_meta().get("version"),
            "trained_on": model_meta().get("data_source"),   # synthetic | live | mixed
        },
        "data_mode": live.data_mode(ZONES_MAP)["mode"],      # demo | live | estimated
        "yolo_model": cv_status,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


# ---------------------------------------------------------------------------
# Zone Catalogue
# ---------------------------------------------------------------------------

@router.get("/zones", dependencies=[Depends(require("view_occupancy"))])
def list_zones():
    result = []
    for z_id, z in ZONES_MAP.items():
        result.append({
            "zone_id": z["id"],
            "name": z["name"],
            "category": z["category"],
            "operational_capacity": z["operational_capacity"],
            "theoretical_capacity": z["theoretical_capacity"],
            "camera_id": z["camera_id"],
            "nearby_landmarks": z.get("nearby_landmarks", [])
        })
    return result


# ---------------------------------------------------------------------------
# Real ML Prediction (single zone)
# ---------------------------------------------------------------------------

@router.post("/predict", response_model=PredictionResponse, dependencies=[Depends(require("view_occupancy"))])
def predict_occupancy(req: PredictionRequest):
    if req.zone_id not in ZONES_MAP:
        raise HTTPException(
            status_code=404,
            detail=f"Zone '{req.zone_id}' not found. Valid zones: {list(ZONES_MAP.keys())}"
        )

    if not 5 <= req.horizon_minutes <= 60:
        raise HTTPException(
            status_code=422,
            detail="horizon_minutes must be between 5 and 60 (interpolated between the trained 15/30/60 horizons)."
        )

    if req.current_count < 0:
        raise HTTPException(status_code=422, detail="current_count must be >= 0.")

    z_meta = ZONES_MAP[req.zone_id]
    cap = z_meta["operational_capacity"]

    ts = None
    if req.timestamp:
        try:
            ts = datetime.fromisoformat(req.timestamp)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=_IST)
            else:
                ts = ts.astimezone(_IST)
        except ValueError:
            raise HTTPException(status_code=422, detail="timestamp must be ISO 8601 format.")

    result = predict_zone(
        zone_id=req.zone_id,
        current_count=req.current_count,
        operational_capacity=cap,
        horizon_minutes=req.horizon_minutes,
        is_exam_day=req.is_exam_day,
        is_class_transition=req.is_class_transition,
        timestamp=ts,
        lag_counts=req.lag_counts,
    )

    confidence = "HIGH_CONFIDENCE_ML" if result["inference_source"] == "xgboost_model" else "HEURISTIC_FALLBACK"
    # (HEURISTIC_FALLBACK now means the normal-day profile, used only when no trained model is loaded)

    return PredictionResponse(
        zone_id=req.zone_id,
        zone_name=z_meta["name"],
        horizon_minutes=req.horizon_minutes,
        predicted_vehicle_count=result["predicted_vehicle_count"],
        predicted_low=result.get("predicted_low"),
        predicted_high=result.get("predicted_high"),
        operational_capacity=cap,
        available_spaces=result["available_spaces"],
        occupancy_rate=result["occupancy_rate"],
        saturation_probability=result["saturation_probability"],
        p_free=result["p_free"],
        status=result["status"],
        confidence_level=confidence,
        lag_source=result.get("lag_source", "estimated"),
        stream_frozen=result.get("stream_frozen", False)
    )


# ---------------------------------------------------------------------------
# Data collection — camera counts in, training data out
# ---------------------------------------------------------------------------

_store: Optional[ObservationStore] = None


def get_store() -> ObservationStore:
    global _store
    if _store is None:
        _store = ObservationStore(os.environ.get("PARKING_DB", DEFAULT_DB))
    return _store


@router.post("/ingest")
def ingest_observation(req: IngestRequest, request: Request, x_api_key: Optional[str] = Header(None)):
    """
    Store one camera's vehicle count. The zone total for the 5-minute bucket is the sum over
    all cameras reporting for that zone, and it is pushed into the rolling history so the next
    forecast uses real lags (`lag_source: observed`).
    Callers authenticate either with the `X-API-Key` header (camera devices; the key is set in
    INGEST_API_KEY) or with a logged-in admin session (camera_control).
    """
    required = os.environ.get("INGEST_API_KEY")
    if x_api_key is not None:
        if not required or not hmac.compare_digest(x_api_key, required):
            raise HTTPException(status_code=401, detail="Invalid X-API-Key.")
    else:
        require("camera_control")(request)
    if req.zone_id not in ZONES_MAP:
        raise HTTPException(status_code=404, detail=f"Zone '{req.zone_id}' not found.")
    try:
        ts = datetime.fromisoformat(req.timestamp) if req.timestamp else datetime.now(_IST)
    except ValueError:
        raise HTTPException(status_code=422, detail="timestamp must be ISO 8601 format.")
    cam = req.camera_id or ZONES_MAP[req.zone_id]["camera_id"]
    res = get_store().record(req.zone_id, cam, req.count, ts, req.source)
    HISTORY.record(req.zone_id, res["bucket"], res["zone_total"])
    live.note_real_data()          # from now on this campus is on real data: no demo feed, fresh forecasts
    cap = ZONES_MAP[req.zone_id]["operational_capacity"]
    return {"zone_id": req.zone_id, "bucket": res["bucket"].isoformat(),
            "zone_total": res["zone_total"], "cameras_reporting": res["cameras_reporting"],
            "occupancy_rate": round(res["zone_total"] / cap, 4),
            "over_capacity": res["zone_total"] > cap}


@router.get("/observations/stats", dependencies=[Depends(require("camera_control"))])
def observation_stats():
    """How much real data has been collected so far."""
    return get_store().stats()


# ---------------------------------------------------------------------------
# Demand forecasting, zone-level analysis, heatmaps
# ---------------------------------------------------------------------------

_demand_model = None


def _get_demand_model():
    global _demand_model
    if _demand_model is None:
        import joblib
        path = "artifacts/pipelines/demand_forecaster.joblib"
        if not os.path.exists(path):
            raise HTTPException(status_code=503, detail="Demand model not trained. Run python -m models.train.")
        _demand_model = joblib.load(path)
    return _demand_model


def _parse_dates(raw: Optional[str], name: str):
    out = set()
    for tok in (raw or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.add(datetime.fromisoformat(tok).date())
        except ValueError:
            raise HTTPException(status_code=422, detail=f"{name}: '{tok}' is not an ISO date (YYYY-MM-DD).")
    return out


@router.get("/demand-forecast", dependencies=[Depends(require("view_analytics"))])
def demand_forecast(
    days: int = Query(7, ge=1, le=14, description="Days ahead"),
    start: Optional[str] = Query(None, description="First day (YYYY-MM-DD, default today IST)"),
    zone_id: Optional[str] = Query(None),
    exam_dates: Optional[str] = Query(None, description="Comma-separated dates to treat as exam days (what-if)"),
    event_dates: Optional[str] = Query(None, description="Comma-separated dates with a campus event (what-if)"),
    holiday_dates: Optional[str] = Query(None, description="Comma-separated dates to treat as holidays (what-if)"),
):
    """
    Calendar-based demand outlook: expected peak time and level, hours above 90% and the
    24-hour profile for every zone and day. The campus calendar (exam windows, holidays and events with
    their venue and expected vehicles) is applied automatically; the *_dates arguments add what-if days.
    """
    if zone_id and zone_id not in ZONES_MAP:
        raise HTTPException(status_code=404, detail=f"Zone '{zone_id}' not found.")
    try:
        first = datetime.fromisoformat(start).date() if start else datetime.now(_IST).date()
    except ValueError:
        raise HTTPException(status_code=422, detail="start must be YYYY-MM-DD.")
    exam, event = _parse_dates(exam_dates, "exam_dates"), _parse_dates(event_dates, "event_dates")
    hol = _parse_dates(holiday_dates, "holiday_dates")
    window_start = datetime.combine(first, datetime.min.time())
    entries = calendar_store.entries_between(window_start, window_start + timedelta(days=days))
    grid = _get_demand_model().forecast_grid(first, days, exam_dates=exam, event_dates=event,
                                             zone_ids=[zone_id] if zone_id else None,
                                             holiday_dates=hol, entries=entries)
    rows = summarise_days(grid)
    for r in rows:
        r["zone_name"] = ZONES_MAP[r["zone_id"]]["name"]
    return {"start": first.isoformat(), "days": days, "zone_id": zone_id,
            "scenario": {"exam_dates": sorted(d.isoformat() for d in exam),
                         "event_dates": sorted(d.isoformat() for d in event),
                         "holiday_dates": sorted(d.isoformat() for d in hol)},
            "calendar_entries_applied": [calendar_store.to_dict(e) for e in entries],
            "forecast": rows}


@router.get("/zone-analysis/{zone_id}", dependencies=[Depends(require("view_occupancy"))])
def get_zone_analysis(zone_id: str):
    """Historical behaviour of one zone: profiles, peak, fill/drain times, saturation share."""
    if zone_id not in ZONES_MAP:
        raise HTTPException(status_code=404, detail=f"Zone '{zone_id}' not found.")
    out = zone_analytics.zone_analysis(zone_id)
    out["zone_name"] = ZONES_MAP[zone_id]["name"]
    out["category"] = ZONES_MAP[zone_id]["category"]
    return out


@router.get("/heatmap", dependencies=[Depends(require("view_analytics"))])
def get_heatmap(
    kind: str = Query("zone_hour", description="zone_hour | zone_week | forecast"),
    zone_id: Optional[str] = Query(None),
    day_type: str = Query("weekday", description="weekday | weekend (zone_hour only)"),
    date: Optional[str] = Query(None, description="YYYY-MM-DD for kind=forecast (default today)"),
):
    """Matrices for heatmaps: historical zone x hour, one zone's weekday x hour, or tomorrow's forecast."""
    try:
        if kind == "forecast":
            day = datetime.fromisoformat(date).date() if date else datetime.now(_IST).date()
            rows = summarise_days(_get_demand_model().forecast_grid(day, 1))
            rows.sort(key=lambda r: -r["peak_occupancy"])
            return {"kind": kind, "date": day.isoformat(), "cols": list(range(24)),
                    "rows": [ZONES_MAP[r["zone_id"]]["name"] for r in rows],
                    "row_ids": [r["zone_id"] for r in rows],
                    "values": [r["hourly"] for r in rows], "data_source": "model_forecast"}
        return zone_analytics.heatmap(kind, zone_id, day_type)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Zone '{zone_id}' not found.")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


# ---------------------------------------------------------------------------
# Explainability — SHAP breakdown of a single prediction
# ---------------------------------------------------------------------------

@router.get("/explain", dependencies=[Depends(require("view_analytics"))])
def explain_prediction(
    zone_id: str = Query(..., description="Zone to explain, e.g. ZONE_BLOCK1"),
    current_count: int = Query(..., ge=0),
    horizon_minutes: int = Query(15),
    is_exam_day: Optional[bool] = Query(None),
    is_class_transition: Optional[bool] = Query(None),
):
    """
    Why did the model predict this? Returns the additive SHAP decomposition:
    prediction = base_value + sum(effect_cars). Positive effects push the
    forecast up (more cars), negative effects push it down.
    """
    if zone_id not in ZONES_MAP:
        raise HTTPException(status_code=404, detail=f"Zone '{zone_id}' not found.")
    if horizon_minutes not in (15, 30, 60):
        raise HTTPException(status_code=422, detail="horizon_minutes must be 15, 30, or 60.")
    if not models_ready()["loaded"]:
        raise HTTPException(status_code=503, detail="Model artifacts not available.")
    from models.explain import explain_row  # imported lazily: SHAP is heavy

    cap = ZONES_MAP[zone_id]["operational_capacity"]
    X, lag_source = _build_feature_row(
        zone_id, current_count, cap, is_exam_day, is_class_transition,
        datetime.now(_IST), history=HISTORY)
    out = explain_row(X, {15: "H15", 30: "H30", 60: "H60"}[horizon_minutes])
    return {"zone_id": zone_id, "horizon_minutes": horizon_minutes,
            "lag_source": lag_source, **out}


# ---------------------------------------------------------------------------
# Recommendations — drives Driver Assistant
# ---------------------------------------------------------------------------

@router.get("/recommendations", response_model=RecommendationResponse)
def get_recommendations(
    destination: Optional[str] = Query(None, description="Campus block you are heading to. Defaults to your home block."),
    horizon_minutes: int = Query(15, ge=1, le=120, description="Minutes until you arrive"),
    arrive_at: Optional[str] = Query(None, description="Arrival clock time HH:MM (next occurrence, campus time) or "
                                                       "YYYY-MM-DDTHH:MM; overrides horizon_minutes"),
    is_exam_day: Optional[bool] = Query(None, description="What-if override; omit to use the campus calendar"),
    is_class_transition: Optional[bool] = Query(None, description="What-if override; omit to use the bell schedule"),
    user: Dict[str, Any] = Depends(require("recommend")),
):
    """
    Nearest-parking recommendations. Without `destination`, ranking starts from the caller's
    home block. A block that has no measured distances of its own is ranked from the adjacent
    reference block (configs/campus_blocks.yaml). Grounds a warden marked CLOSED or FULL are
    listed last and never suggested.
    """
    blocks = get_campus_blocks()
    if destination and destination.strip():
        requested, source = destination.strip(), "selected"
    elif user.get("home_block"):
        requested, source = user["home_block"], "home_block"
    else:
        raise HTTPException(
            status_code=422,
            detail="No destination given and no home block on your profile. "
                   "Pick a destination, or set your home block on the Account page.")
    block = blocks.get(requested)
    if block is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown destination '{requested}'. Valid: {[b['id'] for b in blocks.all()]}")
    destination = block["anchor"]

    now = datetime.now(_IST)
    now_naive = now.replace(tzinfo=None)
    if arrive_at:
        arrival, horizon_minutes = resolve_arrival(arrive_at, now_naive)
    else:
        arrival = now_naive + timedelta(minutes=horizon_minutes)
    all_preds = live.predict_all(ZONES_MAP, now_naive, horizon_minutes, is_exam_day, is_class_transition)
    overrides = active_overrides()

    # Map predictions into the format ParkingRanker expects
    pred_input = {}
    for z_id, p in all_preds.items():
        pred_input[z_id] = {
            "predicted_count":   p["predicted_vehicle_count"],
            "available_spaces":  p["available_spaces"],
            "occupancy_rate":    p["occupancy_rate"],
            "saturation_prob":   p["saturation_probability"],
            "camera_age_seconds": p.get("camera_age_seconds", 30.0),
            "p_free":            p["p_free"] if p["inference_source"] == "xgboost_model" else None,
            "confidence":        p["confidence"],
            "count_source":      p["count_source"],
            "predicted_low":     p["predicted_low"],
            "predicted_high":    p["predicted_high"],
            "forecast_basis":    p["forecast_basis"],
        }
        if z_id in overrides:   # closed / full grounds rank as saturated
            cap = ZONES_MAP[z_id]["operational_capacity"]
            pred_input[z_id].update(predicted_count=cap, available_spaces=0, occupancy_rate=1.0, saturation_prob=1.0,
                                    p_free=0.0)

    ranked = ranker.rank_zones(destination, pred_input, horizon_minutes=horizon_minutes)
    if overrides:
        usable, blocked = [], []
        for z in ranked:
            o = overrides.get(z["zone_id"])
            if o:
                z.update(status=o["status"], available_spaces=0, score=0.0,
                         override_reason=o.get("reason") or f"Marked {o['status'].lower()} by {o['set_by']}")
                blocked.append(z)
            else:
                usable.append(z)
        ranked = usable + blocked

    tip = _arrival_tip(ranked, horizon_minutes, now_naive, is_exam_day, is_class_transition)
    bases = {p["forecast_basis"] for p in all_preds.values()}

    return RecommendationResponse(
        destination=destination,
        horizon_minutes=horizon_minutes,
        recommended_zones=ranked,
        counterfactual_tip=tip,
        requested_block=block["id"],
        requested_block_name=block["name"],
        destination_source=source,
        uses_adjacent_block=not block["is_reference"],
        data_mode=live.data_mode(ZONES_MAP)["mode"],
        # conditions at the ARRIVAL time: that is what the forecast is about
        conditions=cal.context_at(arrival, calendar_store.list_entries()),
        arrival_time=arrival.isoformat(timespec="minutes"),
        forecast_basis="calendar" if bases == {"calendar"} else "live" if bases == {"live"} else "blended",
    )


def resolve_arrival(arrive_at: str, now: datetime):
    """'HH:MM' -> next occurrence (today, or tomorrow if already past); ISO date-time -> as given. Returns (arrival, minutes)."""
    v = arrive_at.strip()
    max_days = load_model_config()["live"]["max_arrival_days"]
    try:
        if len(v) <= 5 and ":" in v:
            hh, mm = (int(x) for x in v.split(":"))
            arrival = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
            if arrival <= now + timedelta(minutes=1):
                arrival += timedelta(days=1)
        else:
            arrival = datetime.fromisoformat(v)
            arrival = arrival.astimezone(_IST).replace(tzinfo=None) if arrival.tzinfo else arrival
    except ValueError:
        raise HTTPException(status_code=422, detail="arrive_at must be HH:MM or YYYY-MM-DDTHH:MM.")
    if arrival <= now:
        raise HTTPException(status_code=422, detail="arrive_at is in the past.")
    if arrival > now + timedelta(days=max_days):
        raise HTTPException(status_code=422, detail=f"arrive_at can be at most {max_days} days ahead.")
    return arrival, max(1, int(round((arrival - now).total_seconds() / 60)))


def _arrival_tip(ranked, horizon_minutes, now, is_exam_day, is_class_transition) -> Optional[str]:
    """
    A tip computed from the forecast itself (no canned claims): if the best ground is unlikely to have
    space, how much later arrival helps; otherwise any campus event or exam affecting the grounds.
    """
    usable = [z for z in ranked if z["status"] not in ("CLOSED", "FULL")]
    if not usable:
        return "Every ground is currently closed or full. Check with the parking office."
    best = usable[0]
    if best.get("p_free") is not None and best["p_free"] < 0.7:
        base = best["p_free"]
        for extra in (15, 30, 45, 60):
            later = live.predict_all(ZONES_MAP, None, horizon_minutes + extra, is_exam_day, is_class_transition)
            p_later = later[best["zone_id"]]["p_free"]
            if p_later >= base + 0.15 and p_later >= 0.6:
                return (f"{best['name']} has a {round(100 * base)}% chance of a free slot if you arrive in "
                        f"{horizon_minutes} min; arriving about {extra} min later raises that to {round(100 * p_later)}%.")
        if base < 0.35:
            return (f"All nearby grounds are likely to be busy at that time; {best['name']} is your best chance "
                    f"({round(100 * base)}% for a free slot). Leave earlier or allow extra walking time.")
    ctx = cal.context_at(now, calendar_store.list_entries())
    if ctx["events_now"]:
        e = ctx["events_now"][0]
        return f"Campus event on: {e['name']}. Grounds near {str(e['venue_block']).replace('_', ' ')} are busier than usual."
    if ctx["events_soon"]:
        e = ctx["events_soon"][0]
        return f"{e['name']} starts soon near {str(e['venue_block']).replace('_', ' ')}; nearby grounds will fill up."
    if ctx["exam_day"]:
        return "Exam day: academic-block grounds fill earlier than usual."
    return None


def _get_valid_destinations():
    destinations = set()
    for z in ZONES_MAP.values():
        destinations.update(z.get("walking_distances_meters", {}).keys())
    return sorted(destinations)


# ---------------------------------------------------------------------------
# Driver Navigation & Anti-Herding Dispatch
# ---------------------------------------------------------------------------

@router.post("/navigate")
@limiter.limit("30/minute")
def record_driver_navigation(request: Request, req: NavigateRequest, user: Dict[str, Any] = Depends(require("recommend"))):
    """
    Registers a driver dispatching towards a specific parking zone.
    Updates the InFlightDispatchTracker to proactively prevent congestion herd collapse.
    """
    if req.zone_id not in ZONES_MAP:
        raise HTTPException(status_code=404, detail="Zone not found.")
    o = active_overrides().get(req.zone_id)
    if o:
        raise HTTPException(status_code=409, detail=f"{ZONES_MAP[req.zone_id]['name']} is marked {o['status'].lower()}"
                                                    f"{': ' + o['reason'] if o.get('reason') else ''}. Pick another ground.")
    cap = ZONES_MAP[req.zone_id]["operational_capacity"]
    active_count = ranker.dispatch_tracker.record_dispatch(req.zone_id, max_capacity=cap)
    return {
        "status": "DISPATCH_REGISTERED",
        "zone_id": req.zone_id,
        "active_in_flight_count": active_count,
        "message": f"Navigation locked to {ZONES_MAP[req.zone_id]['name']}. Spot buffered against herd collapse."
    }


# ---------------------------------------------------------------------------
# Real-Time Campus Snapshot — drives Admin occupancy chart
# ---------------------------------------------------------------------------

@router.get("/snapshot", dependencies=[Depends(require("view_occupancy"))])
def get_campus_snapshot(
    is_exam_day: Optional[bool] = Query(None),
    is_class_transition: Optional[bool] = Query(None),
):
    """
    Every zone right now: the current count (live camera, or labelled demo / estimate), and the forecast
    15 minutes ahead with its 80% range and the chance of a free slot.
    """
    now = datetime.now(_IST)
    all_preds = live.predict_all(ZONES_MAP, None, 15, is_exam_day, is_class_transition)

    zones_data = []
    total_capacity = 0
    total_occupied = 0
    overrides = active_overrides()
    blocked_free = 0   # free space in grounds a warden closed / marked full

    for z_id, z_meta in ZONES_MAP.items():
        pred = all_preds.get(z_id, {})
        cap = z_meta["operational_capacity"]
        occ = pred.get("current_count", 0)          # what is parked now (not the forecast)
        total_capacity += cap
        total_occupied += occ
        override = overrides.get(z_id)
        if override:
            blocked_free += max(0, cap - occ)

        zones_data.append({
            "zone_id": z_id,
            "name": z_meta["name"],
            "camera_id": z_meta["camera_id"],
            "category": z_meta["category"],
            "operational_capacity": cap,
            "predicted_vehicle_count": pred.get("predicted_vehicle_count", occ),     # +15 min
            "predicted_low": pred.get("predicted_low"),
            "predicted_high": pred.get("predicted_high"),
            "available_spaces": 0 if override else pred.get("available_spaces", cap),
            "occupancy_rate": pred.get("occupancy_rate", 0.0),
            "saturation_probability": pred.get("saturation_probability", 0.0),
            "p_free": pred.get("p_free"),
            "current_count": occ,
            "current_occupancy_rate": round(min(1.0, occ / cap), 4) if cap else 0.0,
            "status": override["status"] if override else pred.get("status", "AVAILABLE"),
            "camera_age_seconds": pred.get("camera_age_seconds", 30.0),
            "camera_status": pred.get("camera_status", "live"),      # live | stale | offline | demo
            "count_source": pred.get("count_source"),                # camera | simulated | profile
            "confidence": pred.get("confidence"),
            "stream_frozen": pred.get("stream_frozen", False),
            "override": override,
        })

    campus_occ_rate = round(total_occupied / total_capacity, 4) if total_capacity > 0 else 0.0

    alerts = []
    for z in zones_data:
        base = {"zone_id": z["zone_id"], "name": z["name"]}
        if z["override"]:
            o = z["override"]
            alerts.append({**base, "severity": "warning", "type": f"ZONE_{o['status']}",
                           "message": f"{z['name']} marked {o['status'].lower()} by {o['set_by']}"
                                      f"{': ' + o['reason'] if o.get('reason') else ''}."})
            continue
        if z["occupancy_rate"] >= 0.90:
            alerts.append({**base, "severity": "critical", "type": "SATURATED",
                           "message": f"{z['name']} is forecast at {round(z['occupancy_rate'] * 100)}% of capacity."})
        elif z["saturation_probability"] >= 0.70:
            alerts.append({**base, "severity": "warning", "type": "SATURATION_RISK",
                           "message": f"{z['name']}: {round(z['saturation_probability'] * 100)}% chance of reaching 90% within 15 minutes."})
        if z["camera_status"] == "stale":
            alerts.append({**base, "severity": "warning", "type": "STALE_CAMERA",
                           "message": f"{z['name']}: last camera count is {round(z['camera_age_seconds'] / 60)} min old."})
        elif z["camera_status"] == "offline":
            alerts.append({**base, "severity": "warning", "type": "STALE_CAMERA",
                           "message": f"{z['name']}: no recent camera count; showing the normal-day estimate."})
        if z["stream_frozen"]:
            alerts.append({**base, "severity": "warning", "type": "FROZEN_STREAM",
                           "message": f"{z['name']}: identical counts for 10+ minutes; the camera feed may be frozen."})
    alerts.sort(key=lambda a: a["severity"] != "critical")

    mode = live.data_mode(ZONES_MAP)
    ctx = cal.context_at(now.replace(tzinfo=None), calendar_store.list_entries())
    return {
        "alerts": alerts,
        "timestamp": now.isoformat(),
        "total_capacity": total_capacity,
        "total_occupied": total_occupied,
        "total_available": max(0, total_capacity - total_occupied - blocked_free),
        "campus_occupancy_rate": campus_occ_rate,
        "is_peak_hour": bool(ctx["bell_window"] or campus_occ_rate >= 0.6),
        "data_mode": mode["mode"],
        "zones_by_source": mode["zones_by_source"],
        "calendar_context": ctx,
        "model": {"version": model_meta().get("version"), "trained_on": model_meta().get("data_source")},
        "zones": zones_data,
    }


# ---------------------------------------------------------------------------
# Model Performance Metrics — drives Admin benchmark panel
# ---------------------------------------------------------------------------

@router.get("/metrics", dependencies=[Depends(require("view_analytics"))])
def get_model_metrics():
    """
    Returns real model comparison metrics from artifacts/model_comparison.csv.
    Used by the Admin benchmark chart — no hardcoded values.
    """
    csv_path = "artifacts/model_comparison.csv"
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=503, detail="Model metrics not yet computed. Run models/train.py first.")

    df = pd.read_csv(csv_path)
    ml_status = models_ready()

    # Build structured output for chart rendering
    records = df.to_dict(orient="records")
    for r in records:
        for k, v in r.items():
            if isinstance(v, float) and pd.isna(v):
                r[k] = None  # clean NaN for JSON

    # Filter by Holdout Test split if present
    if "Split" in df.columns:
        xgb_h15 = df[(df["Model"] == "Tuned XGBoost") & (df["Horizon"] == "+15 min") & (df["Split"] == "Holdout Test")]
    else:
        xgb_h15 = df[(df["Model"] == "Tuned XGBoost") & (df["Horizon"] == "+15 min")]

    sat_row = df[df["Model"] == "Saturation Classifier"]
    sat_metrics = {}
    if len(sat_row) > 0:
        row = sat_row.iloc[0]
        sat_metrics = {
            "pr_auc": float(row["PR_AUC"]) if "PR_AUC" in row and pd.notna(row["PR_AUC"]) else None,
            "recall": float(row["Recall"]) if "Recall" in row and pd.notna(row["Recall"]) else None,
            "accuracy": float(row["Accuracy"]) if "Accuracy" in row and pd.notna(row["Accuracy"]) else None,
        }

    sat_path = "artifacts/saturation_metrics.json"
    saturation = json.load(open(sat_path)) if os.path.exists(sat_path) else None
    return {
        "saturation": saturation,
        "models": records,
        "ml_ready": ml_status["loaded"],
        "xgboost_h15": {
            "mae":            float(xgb_h15["MAE"].values[0])            if len(xgb_h15) else None,
            "r2":             float(xgb_h15["R2"].values[0])             if len(xgb_h15) else None,
            "inflection_mae": float(xgb_h15["Inflection_MAE"].values[0]) if len(xgb_h15) and pd.notna(xgb_h15["Inflection_MAE"].values[0]) else None,
        },
        "saturation_classifier": sat_metrics
    }


ALLOWED_FIGURES = {
    "shap_summary_beeswarm.png",
    "shap_waterfall_rush_hour.png",
    "shap_waterfall_steady_state.png",
    "shap_feature_importance_bar.png",
    "residuals_vs_fitted.png",
    "error_distribution_qq.png",
    "error_by_zone_boxplot.png",
    "feature_correlation_heatmap.png",
}

@router.get("/shap-figure/{filename}", dependencies=[Depends(require("view_analytics"))])
def get_shap_figure(filename: str):
    """
    Serves generated SHAP and diagnostic figures safely from artifacts/figures.
    """
    if filename not in ALLOWED_FIGURES:
        raise HTTPException(status_code=404, detail="Figure not found or not in allowed list.")
    
    path = os.path.join("artifacts", "figures", filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Figure {filename} has not been generated yet.")
    
    return FileResponse(path, media_type="image/png")


# ---------------------------------------------------------------------------
# YOLO Vision — Image Upload Detection with ROI & Dual-Model Comparison
# ---------------------------------------------------------------------------

@router.post("/detect-image", dependencies=[Depends(require("camera_control"))])
@limiter.limit("15/minute")
async def detect_image(
    request: Request,
    file: UploadFile = File(...),
    roi: Optional[str] = Form(None),
    model_choice: str = Form("both")
):
    """
    Upload a parking lot image (JPEG/PNG).
    Supports optional ROI polygon/rectangle and compares best.pt with yolo26l.pt.
    Returns annotated images, vehicle counts (ROI vs Total), latency, and comparative metrics.
    """
    allowed_types = {"image/jpeg", "image/jpg", "image/png", "image/bmp", "image/webp"}
    ct = (file.content_type or "").lower()
    if ct and ct not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {ct}. Use JPEG or PNG."
        )

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(image_bytes) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large. Maximum 20 MB.")

    parsed_roi = None
    if roi:
        try:
            raw_roi = json.loads(roi)
            if isinstance(raw_roi, list) and len(raw_roi) >= 3:
                parsed_roi = raw_roi
            elif isinstance(raw_roi, dict) and "polygon" in raw_roi:
                parsed_roi = raw_roi["polygon"]
        except Exception as e:
            logger.warning(f"Failed to parse ROI coordinates: {e}")

    vision = _server_vision()
    if vision is None:
        raise HTTPException(status_code=503, detail=_NO_SERVER_VISION)
    compare_models_from_bytes, detect_from_bytes = vision.compare_models_from_bytes, vision.detect_from_bytes

    if model_choice == "both":
        comp_res = await run_in_threadpool(compare_models_from_bytes, image_bytes, parsed_roi)
        best_m = comp_res.get("best_model", {})
        yolo_m = comp_res.get("yolo26l_model", {})

        target_count = best_m.get("vehicle_count_roi", 0) if parsed_roi else best_m.get("vehicle_count_total", 0)
        best_loaded = best_m.get("error") is None
        yolo_loaded = yolo_m.get("error") is None

        return JSONResponse(content={
            "roi_defined": comp_res.get("roi_defined", False),
            "roi_polygon": parsed_roi,
            "best_model": best_m,
            "yolo26l_model": yolo_m,
            "comparison": comp_res.get("comparison", {}),
            # Root fields for standard contract & backwards compatibility
            "vehicle_count": target_count,
            "vehicle_count_total": best_m.get("vehicle_count_total", 0),
            "vehicle_count_roi": best_m.get("vehicle_count_roi", 0),
            "detections": best_m.get("detections", []),
            "annotated_image_base64": best_m.get("annotated_image_base64", ""),
            "inference_time_ms": best_m.get("inference_time_ms", 0.0),
            "model_loaded": best_loaded and yolo_loaded,
            "error": comp_res.get("error", None),
            "filename": file.filename or "unknown",
        })
    else:
        single_res = await run_in_threadpool(detect_from_bytes, image_bytes, parsed_roi, model_choice)
        target_count = single_res.get("vehicle_count_roi", 0) if parsed_roi else single_res.get("vehicle_count_total", 0)
        return JSONResponse(content={
            "roi_defined": parsed_roi is not None,
            "roi_polygon": parsed_roi,
            "vehicle_count": target_count,
            "vehicle_count_total": single_res.get("vehicle_count_total", 0),
            "vehicle_count_roi": single_res.get("vehicle_count_roi", 0),
            "detections": single_res.get("detections", []),
            "annotated_image_base64": single_res.get("annotated_image_base64", ""),
            "inference_time_ms": single_res.get("inference_time_ms", 0.0),
            "model_loaded": single_res.get("error") is None,
            "error": single_res.get("error", None),
            "filename": file.filename or "unknown",
        })


# ---------------------------------------------------------------------------
# YOLO Vision — Live Webcam Frame
# ---------------------------------------------------------------------------

@router.post("/detect-frame", dependencies=[Depends(require("camera_control"))])
@limiter.limit("60/minute")
async def detect_frame(request: Request, file: UploadFile = File(...)):
    """
    Accept a single webcam frame (JPEG blob from browser canvas.toBlob).
    Returns annotated frame (base64 JPEG) + count + detections.
    """
    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty frame received.")
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Frame too large. Maximum 10 MB.")

    vision = _server_vision()
    if vision is None:
        raise HTTPException(status_code=503, detail=_NO_SERVER_VISION)
    result = await run_in_threadpool(vision.detect_from_bytes, image_bytes, None, "best")

    total_count = result.get("vehicle_count_total", result.get("vehicle_count", 0))

    return JSONResponse(content={
        "vehicle_count":          total_count,
        "vehicle_count_total":    total_count,
        "detections":             result.get("detections", []),
        "annotated_image_base64": result.get("annotated_image_base64", ""),
        "inference_time_ms":      result.get("inference_time_ms", 0.0),
        "model_loaded":           result.get("error") is None,
        "error":                  result.get("error"),
    })
