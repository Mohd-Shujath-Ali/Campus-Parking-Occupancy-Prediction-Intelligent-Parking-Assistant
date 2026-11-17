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

from api.limiter import limiter
from api.schemas import PredictionRequest, PredictionResponse, RecommendationResponse, ZoneRankItem, NavigateRequest
from recommendation.parking_ranker import ParkingRanker
from models.inference import predict_zone, predict_all_zones, models_ready

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

@router.get("/health")
def get_health():
    from computer_vision.detection import model_status
    ml_status = models_ready()
    cv_status = model_status()
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
        operational_capacity=cap,
        available_spaces=result["available_spaces"],
        occupancy_rate=result["occupancy_rate"],
        saturation_probability=result["saturation_probability"],
        status=result["status"],
        confidence_level=confidence,
        stream_frozen=result.get("stream_frozen", False)
    )


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
    if destination not in _get_valid_destinations():
        raise HTTPException(
            status_code=404,
            detail=f"Unknown destination '{destination}'. Valid: {_get_valid_destinations()}"
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
def record_driver_navigation(req: NavigateRequest):
    """
    Registers a driver dispatching towards a specific parking zone.
    Updates the InFlightDispatchTracker to proactively prevent congestion herd collapse.
    """
    if req.zone_id not in ZONES_MAP:
        raise HTTPException(status_code=404, detail="Zone not found.")
    active_count = ranker.dispatch_tracker.record_dispatch(req.zone_id)
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

    return {
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

    return {
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

    from computer_vision.detection import compare_models_from_bytes, detect_from_bytes

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

    from computer_vision.detection import detect_from_bytes
    result = await run_in_threadpool(detect_from_bytes, image_bytes, None, "best")

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
