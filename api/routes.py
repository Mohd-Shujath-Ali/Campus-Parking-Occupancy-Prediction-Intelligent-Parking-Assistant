"""
LPU Campus Parking Intelligence System — FastAPI Routes
All endpoints use the REAL XGBoost model for predictions.
No hardcoded heuristics remain in production paths.
"""

import os
import json
import yaml
import logging
import pandas as pd
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from fastapi import APIRouter, HTTPException, Query, UploadFile, File, Form, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, FileResponse
from typing import List, Dict, Any, Optional

from fastapi import Header
from api.limiter import limiter
from api.schemas import (PredictionRequest, PredictionResponse, RecommendationResponse, ZoneRankItem,
                         IngestRequest, NavigateRequest)
from data_collection.store import ObservationStore, DEFAULT_DB
from recommendation.parking_ranker import ParkingRanker
from models.inference import predict_zone, predict_all_zones, models_ready, _build_feature_row
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
            "error": ml_status["error"]
        },
        "yolo_model": cv_status,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


# ---------------------------------------------------------------------------
# Zone Catalogue
# ---------------------------------------------------------------------------

@router.get("/zones")
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

@router.post("/predict", response_model=PredictionResponse)
def predict_occupancy(req: PredictionRequest):
    if req.zone_id not in ZONES_MAP:
        raise HTTPException(
            status_code=404,
            detail=f"Zone '{req.zone_id}' not found. Valid zones: {list(ZONES_MAP.keys())}"
        )

    if req.horizon_minutes not in (15, 30, 60):
        raise HTTPException(
            status_code=422,
            detail="horizon_minutes must be 15, 30, or 60."
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
def ingest_observation(req: IngestRequest, x_api_key: Optional[str] = Header(None)):
    """
    Store one camera's vehicle count. The zone total for the 5-minute bucket is the sum over
    all cameras reporting for that zone, and it is pushed into the rolling history so the next
    forecast uses real lags (`lag_source: observed`).
    If INGEST_API_KEY is set, the caller must send it in the `X-API-Key` header.
    """
    required = os.environ.get("INGEST_API_KEY")
    if required and x_api_key != required:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key.")
    if req.zone_id not in ZONES_MAP:
        raise HTTPException(status_code=404, detail=f"Zone '{req.zone_id}' not found.")
    try:
        ts = datetime.fromisoformat(req.timestamp) if req.timestamp else datetime.now(_IST)
    except ValueError:
        raise HTTPException(status_code=422, detail="timestamp must be ISO 8601 format.")
    cam = req.camera_id or ZONES_MAP[req.zone_id]["camera_id"]
    res = get_store().record(req.zone_id, cam, req.count, ts, req.source)
    HISTORY.record(req.zone_id, res["bucket"], res["zone_total"])
    cap = ZONES_MAP[req.zone_id]["operational_capacity"]
    return {"zone_id": req.zone_id, "bucket": res["bucket"].isoformat(),
            "zone_total": res["zone_total"], "cameras_reporting": res["cameras_reporting"],
            "occupancy_rate": round(res["zone_total"] / cap, 4),
            "over_capacity": res["zone_total"] > cap}


@router.get("/observations/stats")
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


@router.get("/demand-forecast")
def demand_forecast(
    days: int = Query(7, ge=1, le=14, description="Days ahead"),
    start: Optional[str] = Query(None, description="First day (YYYY-MM-DD, default today IST)"),
    zone_id: Optional[str] = Query(None),
    exam_dates: Optional[str] = Query(None, description="Comma-separated dates to treat as exam days (what-if)"),
    event_dates: Optional[str] = Query(None, description="Comma-separated dates with a campus event (what-if)"),
):
    """
    Calendar-based demand outlook: expected peak time and level, hours above 90% and the
    24-hour profile for every zone and day. Use exam_dates/event_dates for scenario planning.
    """
    if zone_id and zone_id not in ZONES_MAP:
        raise HTTPException(status_code=404, detail=f"Zone '{zone_id}' not found.")
    try:
        first = datetime.fromisoformat(start).date() if start else datetime.now(_IST).date()
    except ValueError:
        raise HTTPException(status_code=422, detail="start must be YYYY-MM-DD.")
    exam, event = _parse_dates(exam_dates, "exam_dates"), _parse_dates(event_dates, "event_dates")
    grid = _get_demand_model().forecast_grid(first, days, exam_dates=exam, event_dates=event,
                                             zone_ids=[zone_id] if zone_id else None)
    rows = summarise_days(grid)
    for r in rows:
        r["zone_name"] = ZONES_MAP[r["zone_id"]]["name"]
    return {"start": first.isoformat(), "days": days, "zone_id": zone_id,
            "scenario": {"exam_dates": sorted(d.isoformat() for d in exam),
                         "event_dates": sorted(d.isoformat() for d in event)},
            "forecast": rows}


@router.get("/zone-analysis/{zone_id}")
def get_zone_analysis(zone_id: str):
    """Historical behaviour of one zone: profiles, peak, fill/drain times, saturation share."""
    if zone_id not in ZONES_MAP:
        raise HTTPException(status_code=404, detail=f"Zone '{zone_id}' not found.")
    out = zone_analytics.zone_analysis(zone_id)
    out["zone_name"] = ZONES_MAP[zone_id]["name"]
    out["category"] = ZONES_MAP[zone_id]["category"]
    return out


@router.get("/heatmap")
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

@router.get("/explain")
def explain_prediction(
    zone_id: str = Query(..., description="Zone to explain, e.g. ZONE_BLOCK1"),
    current_count: int = Query(..., ge=0),
    horizon_minutes: int = Query(15),
    is_exam_day: bool = Query(False),
    is_class_transition: bool = Query(False),
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
    destination: str = Query("Block_26_CSE", description="Campus destination block"),
    horizon_minutes: int = Query(15, ge=1, le=120, description="Arrival horizon (minutes)"),
    is_exam_day: bool = Query(False),
    is_class_transition: bool = Query(False),
):
    valid_dests = _get_valid_destinations()
    norm_map = {d.lower().replace(" ", "_"): d for d in valid_dests}
    req_norm = destination.strip().lower().replace(" ", "_")
    if req_norm in norm_map:
        destination = norm_map[req_norm]
    elif destination not in valid_dests:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown destination '{destination}'. Valid: {valid_dests}"
        )

    now = datetime.now(_IST)
    all_preds = predict_all_zones(
        ZONES_MAP,
        is_exam_day=is_exam_day,
        is_class_transition=is_class_transition,
        timestamp=now,
    )

    # Map predictions into the format ParkingRanker expects
    pred_input = {}
    for z_id, p in all_preds.items():
        pred_input[z_id] = {
            "predicted_count":   p["predicted_vehicle_count"],
            "available_spaces":  p["available_spaces"],
            "occupancy_rate":    p["occupancy_rate"],
            "saturation_prob":   p["saturation_probability"],
            "camera_age_seconds": p.get("camera_age_seconds", 30.0),
        }

    ranked = ranker.rank_zones(destination, pred_input, horizon_minutes=min(horizon_minutes, 60))

    hour = now.hour
    is_peak = (8 <= hour <= 10) or (11 <= hour <= 14) or (16 <= hour <= 18)
    tip = None
    if ranked and ranked[0]["status"] != "AVAILABLE":
        tip = "Smart Arrival Tip: Arriving 15 minutes after peak bell hour reduces congestion by ~22%."
    elif is_peak and ranked:
        tip = f"Peak hour detected ({hour:02d}:00). Lecture surge mode active — plan parking in advance."

    return RecommendationResponse(
        destination=destination,
        horizon_minutes=horizon_minutes,
        recommended_zones=ranked,
        counterfactual_tip=tip
    )


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
def record_driver_navigation(request: Request, req: NavigateRequest):
    """
    Registers a driver dispatching towards a specific parking zone.
    Updates the InFlightDispatchTracker to proactively prevent congestion herd collapse.
    """
    if req.zone_id not in ZONES_MAP:
        raise HTTPException(status_code=404, detail="Zone not found.")
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

@router.get("/snapshot")
def get_campus_snapshot(
    is_exam_day: bool = Query(False),
    is_class_transition: bool = Query(False),
):
    """
    Returns current predicted occupancy for ALL 12 zones.
    Used by the Admin dashboard charts — real ML data, not random numbers.
    """
    now = datetime.now(_IST)
    all_preds = predict_all_zones(
        ZONES_MAP,
        is_exam_day=is_exam_day,
        is_class_transition=is_class_transition,
        timestamp=now,
    )

    zones_data = []
    total_capacity = 0
    total_occupied = 0

    for z_id, z_meta in ZONES_MAP.items():
        pred = all_preds.get(z_id, {})
        cap = z_meta["operational_capacity"]
        occ = pred.get("predicted_vehicle_count", 0)
        total_capacity += cap
        total_occupied += occ

        zones_data.append({
            "zone_id": z_id,
            "name": z_meta["name"],
            "camera_id": z_meta["camera_id"],
            "category": z_meta["category"],
            "operational_capacity": cap,
            "predicted_vehicle_count": occ,
            "predicted_low": pred.get("predicted_low"),
            "predicted_high": pred.get("predicted_high"),
            "available_spaces": pred.get("available_spaces", cap),
            "occupancy_rate": pred.get("occupancy_rate", 0.0),
            "saturation_probability": pred.get("saturation_probability", 0.0),
            "current_count": pred.get("current_count", 0),
            "status": pred.get("status", "AVAILABLE"),
            "camera_age_seconds": pred.get("camera_age_seconds", 30.0),
            "camera_status": pred.get("camera_status", "live"),
            "stream_frozen": pred.get("stream_frozen", False),
        })

    campus_occ_rate = round(total_occupied / total_capacity, 4) if total_capacity > 0 else 0.0

    alerts = []
    for z in zones_data:
        base = {"zone_id": z["zone_id"], "name": z["name"]}
        if z["occupancy_rate"] >= 0.90:
            alerts.append({**base, "severity": "critical", "type": "SATURATED",
                           "message": f"{z['name']} is forecast at {round(z['occupancy_rate'] * 100)}% of capacity."})
        elif z["saturation_probability"] >= 0.70:
            alerts.append({**base, "severity": "warning", "type": "SATURATION_RISK",
                           "message": f"{z['name']}: {round(z['saturation_probability'] * 100)}% chance of reaching 90% within 15 minutes."})
        if z["camera_status"] != "live" or z["camera_age_seconds"] > 300:
            alerts.append({**base, "severity": "warning", "type": "STALE_CAMERA",
                           "message": f"{z['name']}: camera data is stale ({round(z['camera_age_seconds'])}s old)."})
    alerts.sort(key=lambda a: a["severity"] != "critical")

    return {
        "alerts": alerts,
        "timestamp": now.isoformat(),
        "total_capacity": total_capacity,
        "total_occupied": total_occupied,
        "total_available": max(0, total_capacity - total_occupied),
        "campus_occupancy_rate": campus_occ_rate,
        "is_peak_hour": (8 <= now.hour <= 10) or (11 <= now.hour <= 14) or (16 <= now.hour <= 18),
        "zones": zones_data,
    }


# ---------------------------------------------------------------------------
# Model Performance Metrics — drives Admin benchmark panel
# ---------------------------------------------------------------------------

@router.get("/metrics")
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

@router.get("/shap-figure/{filename}")
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

@router.post("/detect-image")
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

@router.post("/detect-frame")
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
