"""
LPU Campus Parking - Real-Time ML Inference Engine
Wraps the trained XGBoost multi-horizon forecaster and saturation classifier
so the API can serve live predictions from actual sensor/simulated inputs.
"""

import os
import hashlib
import logging
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Dict, Any, Optional
from pathlib import Path

from features.feature_pipeline import FEATURE_COLS, zone_code
from models.history import HISTORY, ZoneHistory

logger = logging.getLogger(__name__)
_IST = ZoneInfo("Asia/Kolkata")


# ---------------------------------------------------------------------------
# Model loader (singleton pattern — loaded once at startup)
# ---------------------------------------------------------------------------
_xgb_model = None
_clf_model = None
_int_model = None
_load_error: Optional[str] = None


def _load_models():
    global _xgb_model, _clf_model, _int_model, _load_error
    if _xgb_model is not None:
        return True
    try:
        xgb_path = Path("artifacts/pipelines/xgb_multi_horizon.joblib")
        clf_path = Path("artifacts/pipelines/saturation_classifier.joblib")
        if not xgb_path.exists() or not clf_path.exists():
            _load_error = f"Model artifacts not found at {xgb_path} or {clf_path}"
            return False
        _xgb_model = joblib.load(xgb_path)
        _clf_model = joblib.load(clf_path)
        int_path = Path("artifacts/pipelines/xgb_intervals.joblib")
        _int_model = joblib.load(int_path) if int_path.exists() else None  # optional
        logger.info("XGBoost forecaster and saturation classifier loaded.")
        return True
    except Exception as e:
        _load_error = str(e)
        logger.error(f"Failed to load ML models: {e}")
        return False


def models_ready() -> Dict[str, Any]:
    ok = _load_models()
    return {"loaded": ok, "error": _load_error}


# ---------------------------------------------------------------------------
# Feature construction from live API inputs
# ---------------------------------------------------------------------------

def _build_feature_row(
    zone_id: str,
    current_count: int,
    operational_capacity: int,
    is_exam_day: bool,
    is_class_transition: bool,
    timestamp: Optional[datetime] = None,
    lag_counts: Optional[Dict[str, int]] = None,  # {'15m':x, '30m':x, '45m':x, '60m':x}
    history: Optional[ZoneHistory] = None,
    is_campus_event: bool = False,
):
    """
    Build the feature vector in the exact schema used for training
    (features.feature_pipeline.FEATURE_COLS).

    Lag sources, in priority order: explicit ``lag_counts`` > rolling
    ``history`` buffer > current count (flagged as "estimated").
    Returns (DataFrame, lag_source) where lag_source is "observed",
    "partial" or "estimated".
    """
    if timestamp is None:
        timestamp = datetime.now(_IST)

    hour_frac = timestamp.hour + timestamp.minute / 60.0
    dow = timestamp.weekday()  # 0=Mon ... 6=Sun
    occ_ratio = current_count / operational_capacity if operational_capacity > 0 else 0.0

    lags: Dict[int, Optional[int]] = {}
    for m in (15, 30, 45, 60):
        v = None
        if lag_counts and f"{m}m" in lag_counts:
            v = lag_counts[f"{m}m"]
        elif history is not None:
            v = history.lag(zone_id, timestamp, m)
        lags[m] = v
    found = sum(v is not None for v in lags.values())
    lag_source = "observed" if found == 4 else "partial" if found else "estimated"
    lag_val = {m: (v if v is not None else current_count) for m, v in lags.items()}

    # Rolling stats: training uses the 5-minute samples t-5..t-W (current step
    # excluded, pandas sample std). Use the same window when history exists.
    def _roll(w: int):
        vals = history.window(zone_id, timestamp, w) if history is not None else []
        if not vals:  # fall back to whatever lags we have
            vals = [lag_val[m] for m in (15, 30, 45, 60) if m <= w] or [current_count]
        arr = np.asarray(vals, dtype=float)
        std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        return float(arr.mean()), std

    rm30, rs30 = _roll(30)
    rm60, rs60 = _roll(60)

    row = {
        'vehicle_count':        current_count,
        'operational_capacity': operational_capacity,
        'occupancy_ratio':      occ_ratio,
        'zone_code':            zone_code(zone_id),
        'hour_sin':             np.sin(2 * np.pi * hour_frac / 24.0),
        'hour_cos':             np.cos(2 * np.pi * hour_frac / 24.0),
        'day_sin':              np.sin(2 * np.pi * dow / 7.0),
        'day_cos':              np.cos(2 * np.pi * dow / 7.0),
        'is_weekend':           int(dow >= 5),
        'is_exam_day':          int(is_exam_day),
        'is_campus_event':      int(is_campus_event),
        'is_class_transition':  int(is_class_transition),
        'lag_15m':              lag_val[15],
        'lag_30m':              lag_val[30],
        'lag_45m':              lag_val[45],
        'lag_60m':              lag_val[60],
        'delta_15m':            current_count - lag_val[15],
        'delta_30m':            current_count - lag_val[30],
        'rolling_mean_30m':     rm30,
        'rolling_std_30m':      rs30,
        'rolling_mean_60m':     rm60,
        'rolling_std_60m':      rs60,
    }
    return pd.DataFrame([row], columns=FEATURE_COLS), lag_source


# ---------------------------------------------------------------------------
# Public inference API
# ---------------------------------------------------------------------------

def predict_zone(
    zone_id: str,
    current_count: int,
    operational_capacity: int,
    horizon_minutes: int = 15,
    is_exam_day: bool = False,
    is_class_transition: bool = False,
    timestamp: Optional[datetime] = None,
    lag_counts: Optional[Dict[str, int]] = None,
    is_campus_event: bool = False,
    history: Optional[ZoneHistory] = HISTORY,
    record: bool = True,
) -> Dict[str, Any]:
    """
    Runs the real XGBoost multi-horizon forecaster on the given zone state.
    Returns predicted count, available spaces, occupancy rate, saturation prob.
    Falls back to heuristic if model is unavailable.
    """
    _load_models()
    if timestamp is None:
        timestamp = datetime.now(_IST)
    lag_source = "estimated"
    low = high = None

    horizon_key_map = {15: "H15", 30: "H30", 60: "H60"}
    h_key = horizon_key_map.get(horizon_minutes, "H15")

    if _xgb_model is None or _clf_model is None:
        # Heuristic fallback (disclosed to caller)
        surge = 1.15 if is_class_transition else 1.02
        exam  = 1.10 if is_exam_day else 1.0
        pred_count = int(min(operational_capacity * 1.15, max(0, current_count * surge * exam)))
        low = high = None
        sat_proba  = 0.5
        source     = "heuristic_fallback"
    else:
        X, lag_source = _build_feature_row(
            zone_id, current_count, operational_capacity,
            is_exam_day, is_class_transition, timestamp, lag_counts,
            history=history, is_campus_event=is_campus_event,
        )
        preds = _xgb_model.predict(X)
        pred_count = int(max(0, min(operational_capacity * 1.20, round(preds[h_key][0]))))
        sat_proba  = float(_clf_model.predict_proba(X)[0])
        source     = "xgboost_model"
        if _int_model is not None:
            lo, hi = _int_model.predict(X, h_key, point=np.array([pred_count], dtype=float))
            low = int(max(0, np.floor(lo[0])))
            high = int(min(operational_capacity * 1.20, np.ceil(hi[0])))

    # Remember this observation so later calls get real lags
    if record and history is not None:
        history.record(zone_id, timestamp, current_count)

    avail      = max(0, operational_capacity - pred_count)
    # Cap occ_rate at 1.0 — a zone cannot exceed 100% by definition
    occ_rate   = round(min(1.0, pred_count / operational_capacity), 4) if operational_capacity > 0 else 0.0
    status     = ("SATURATED" if occ_rate >= 0.90 else
                  "FILLING_FAST" if occ_rate >= 0.75 else "AVAILABLE")

    return {
        "predicted_vehicle_count": pred_count,
        # 80% conformal prediction interval (None when the heuristic fallback is used)
        "predicted_low":           low,
        "predicted_high":          high,
        "operational_capacity":    operational_capacity,
        "available_spaces":        avail,
        "occupancy_rate":          occ_rate,
        "saturation_probability":  round(sat_proba, 4),
        "status":                  status,
        "inference_source":        source,
        # "observed": all lags from real history; "partial"/"estimated": some
        # or all lags were filled with the current count (lower reliability)
        "lag_source":              lag_source,
    }


def predict_all_zones(
    zones_map: Dict[str, Any],
    is_exam_day: bool = False,
    is_class_transition: bool = False,
    timestamp: Optional[datetime] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Predict occupancy for every zone simultaneously, using the latest
    simulated sensor count as the input current_count.
    Returns zone_id -> prediction dict.
    """
    if timestamp is None:
        timestamp = datetime.now(_IST)

    hour = timestamp.hour + timestamp.minute / 60.0
    is_peak = (8.0 <= hour <= 10.0) or (11.0 <= hour <= 14.0) or (16.0 <= hour <= 18.0)

    results = {}
    for z_id, z_meta in zones_map.items():
        cap = z_meta["operational_capacity"]
        # Realistic sensor simulation: occupancy varies by time of day + zone type
        base_occ = 0.72 if is_peak else 0.38
        # Add zone-specific noise seed (deterministic so it's stable per refresh window)
        zone_seed = int(hashlib.md5(f"{z_id}{timestamp.strftime('%Y%m%d%H%M')}".encode()).hexdigest(), 16) % 1000
        rng_offset = (zone_seed % 200 - 100) / 1000.0  # ±0.10 offset
        sim_occ = max(0.05, min(0.98, base_occ + rng_offset +
                                (0.08 if is_class_transition else 0.0) +
                                (0.06 if is_exam_day else 0.0)))
        sim_count = int(cap * sim_occ)

        pred = predict_zone(
            z_id, sim_count, cap,
            horizon_minutes=15,
            is_exam_day=is_exam_day,
            is_class_transition=is_class_transition,
            timestamp=timestamp,
            history=None,    # simulated counts must not mix with real history
            record=False,
        )
        pred["current_count"] = sim_count
        pred["camera_age_seconds"] = float(zone_seed % 60)  # simulated age 0-60s
        pred["camera_status"] = "live"
        pred["count_source"] = "simulated"  # demo mode: not a real detection
        results[z_id] = pred

    return results
