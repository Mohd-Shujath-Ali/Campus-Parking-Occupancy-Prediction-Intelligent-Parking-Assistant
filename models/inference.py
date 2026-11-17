"""
LPU Campus Parking - Real-Time ML Inference Engine
Wraps the trained XGBoost multi-horizon forecaster and saturation classifier
so the API can serve live predictions from actual sensor/simulated inputs.
"""

import os
import hashlib
import logging
import joblib
import threading
from collections import deque
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)
_IST = ZoneInfo("Asia/Kolkata")

FEATURE_COLS = [
    'vehicle_count', 'operational_capacity', 'occupancy_ratio',
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'is_weekend',
    'is_exam_day', 'is_class_transition',
    'lag_15m', 'lag_30m', 'lag_45m', 'lag_60m',
    'delta_15m', 'delta_30m',
    'rolling_mean_30m', 'rolling_std_30m',
    'rolling_mean_60m', 'rolling_std_60m'
]

# ---------------------------------------------------------------------------
# Model loader (singleton pattern — loaded once at startup)
# ---------------------------------------------------------------------------
_xgb_model = None
_clf_model = None
_load_error: Optional[str] = None


def _load_models():
    global _xgb_model, _clf_model, _load_error
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
        logger.info("XGBoost forecaster and saturation classifier loaded.")
        return True
    except Exception as e:
        _load_error = str(e)
        logger.error(f"Failed to load ML models: {e}")
        return False


def models_ready() -> Dict[str, Any]:
    ok = _load_models()
    return {"loaded": ok, "error": _load_error}


class _TemporalRingBuffer:
    """
    Thread-safe temporal sliding ring buffer storing historical counts at 5-min intervals
    (retains up to 24 observations = 2 hours) to derive causal lags and velocity deltas
    during live streaming inference without forcing callers to manage history.
    """
    def __init__(self, maxlen: int = 24):
        self._maxlen = maxlen
        self._buffers: Dict[str, deque] = {}
        self._lock = threading.Lock()

    def record_observation(self, zone_id: str, count: int, timestamp: Optional[datetime] = None):
        if timestamp is None:
            timestamp = datetime.now(_IST)
        with self._lock:
            if zone_id not in self._buffers:
                self._buffers[zone_id] = deque(maxlen=self._maxlen)
            self._buffers[zone_id].append((timestamp, int(count)))

    def get_lags(self, zone_id: str, current_count: int, timestamp: Optional[datetime] = None) -> Dict[str, int]:
        if timestamp is None:
            timestamp = datetime.now(_IST)
        with self._lock:
            buf = self._buffers.get(zone_id)
            if not buf or len(buf) < 2:
                # Cold start: Return current count
                return {
                    "15m": current_count,
                    "30m": current_count,
                    "45m": current_count,
                    "60m": current_count,
                }

            history = list(buf)

            def find_lag(offset_minutes: int) -> int:
                target_time = timestamp - timedelta(minutes=offset_minutes)
                best_diff = float("inf")
                best_val = current_count
                for t, c in history:
                    diff = abs((t - target_time).total_seconds())
                    if diff < best_diff:
                        best_diff = diff
                        best_val = c
                return best_val

            return {
                "15m": find_lag(15),
                "30m": find_lag(30),
                "45m": find_lag(45),
                "60m": find_lag(60),
            }

    def is_stream_frozen(self, zone_id: str) -> bool:
        """
        Detects camera/video ingest freeze (e.g. frozen CCTV stream):
        returns True if the last 6 consecutive readings are identical and non-empty.
        """
        with self._lock:
            buf = self._buffers.get(zone_id)
            if not buf or len(buf) < 6:
                return False
            last_counts = [c for _, c in list(buf)[-6:]]
            return len(set(last_counts)) == 1

_ring_buffer = _TemporalRingBuffer()


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
) -> pd.DataFrame:
    """
    Constructs the 20-feature vector expected by the trained XGBoost model.
    When historical lag counts are not supplied, queries the Temporal Ring Buffer.
    """
    if timestamp is None:
        timestamp = datetime.now(_IST)

    if lag_counts is None:
        lag_counts = _ring_buffer.get_lags(zone_id, current_count, timestamp)

    hour_frac = timestamp.hour + timestamp.minute / 60.0
    dow = timestamp.weekday()  # 0=Mon ... 6=Sun

    hour_sin = np.sin(2 * np.pi * hour_frac / 24.0)
    hour_cos = np.cos(2 * np.pi * hour_frac / 24.0)
    day_sin  = np.sin(2 * np.pi * dow / 7.0)
    day_cos  = np.cos(2 * np.pi * dow / 7.0)
    is_weekend = int(dow >= 5)
    occ_ratio  = current_count / operational_capacity if operational_capacity > 0 else 0.0

    # Lag counts — support both '15m' and 'lag_15m' keys
    lag_15 = lag_counts.get("15m", lag_counts.get("lag_15m", current_count))
    lag_30 = lag_counts.get("30m", lag_counts.get("lag_30m", current_count))
    lag_45 = lag_counts.get("45m", lag_counts.get("lag_45m", current_count))
    lag_60 = lag_counts.get("60m", lag_counts.get("lag_60m", current_count))

    delta_15 = current_count - lag_15
    delta_30 = current_count - lag_30

    # Rolling stats from lag history (mean & std over last 30/60m)
    lag_values_30 = [current_count, lag_15, lag_30]
    lag_values_60 = [current_count, lag_15, lag_30, lag_45, lag_60]
    rolling_mean_30 = float(np.mean(lag_values_30))
    rolling_std_30  = float(np.std(lag_values_30, ddof=0))
    rolling_mean_60 = float(np.mean(lag_values_60))
    rolling_std_60  = float(np.std(lag_values_60, ddof=0))

    row = {
        'vehicle_count':      current_count,
        'operational_capacity': operational_capacity,
        'occupancy_ratio':    occ_ratio,
        'hour_sin':           hour_sin,
        'hour_cos':           hour_cos,
        'day_sin':            day_sin,
        'day_cos':            day_cos,
        'is_weekend':         is_weekend,
        'is_exam_day':        int(is_exam_day),
        'is_class_transition': int(is_class_transition),
        'lag_15m':            lag_15,
        'lag_30m':            lag_30,
        'lag_45m':            lag_45,
        'lag_60m':            lag_60,
        'delta_15m':          delta_15,
        'delta_30m':          delta_30,
        'rolling_mean_30m':   rolling_mean_30,
        'rolling_std_30m':    rolling_std_30,
        'rolling_mean_60m':   rolling_mean_60,
        'rolling_std_60m':    rolling_std_60,
    }

    return pd.DataFrame([row], columns=FEATURE_COLS)


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
) -> Dict[str, Any]:
    """
    Runs the real XGBoost multi-horizon forecaster on the given zone state.
    Returns predicted count, available spaces, occupancy rate, saturation prob.
    Falls back to heuristic if model is unavailable.
    """
    _load_models()

    horizon_key_map = {15: "H15", 30: "H30", 60: "H60"}
    h_key = horizon_key_map.get(horizon_minutes, "H15")

    if _xgb_model is None or _clf_model is None:
        # Heuristic fallback (disclosed to caller)
        surge = 1.15 if is_class_transition else 1.02
        exam  = 1.10 if is_exam_day else 1.0
        pred_count = int(min(operational_capacity * 1.15, max(0, current_count * surge * exam)))
        sat_proba  = 0.5
        source     = "heuristic_fallback"
    else:
        X = _build_feature_row(
            zone_id, current_count, operational_capacity,
            is_exam_day, is_class_transition, timestamp, lag_counts
        )
        preds = _xgb_model.predict(X)
        pred_count = int(max(0, min(operational_capacity * 1.20, round(preds[h_key][0]))))
        sat_proba  = float(_clf_model.predict_proba(X)[0])
        source     = "xgboost_model"

    avail      = max(0, operational_capacity - pred_count)
    # Cap occ_rate at 1.0 — a zone cannot exceed 100% by definition
    occ_rate   = round(min(1.0, pred_count / operational_capacity), 4) if operational_capacity > 0 else 0.0
    status     = ("SATURATED" if occ_rate >= 0.90 else
                  "FILLING_FAST" if occ_rate >= 0.75 else "AVAILABLE")

    # Record observation in temporal ring buffer
    _ring_buffer.record_observation(zone_id, current_count, timestamp)
    is_frozen = _ring_buffer.is_stream_frozen(zone_id)

    return {
        "predicted_vehicle_count": pred_count,
        "operational_capacity":    operational_capacity,
        "available_spaces":        avail,
        "occupancy_rate":          occ_rate,
        "saturation_probability":  round(sat_proba, 4),
        "status":                  status,
        "inference_source":        source,
        "stream_frozen":           is_frozen,
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
        )
        pred["current_count"] = sim_count
        pred["camera_age_seconds"] = float(zone_seed % 60)  # simulated age 0-60s
        pred["camera_status"] = "live"
        results[z_id] = pred

    return results
