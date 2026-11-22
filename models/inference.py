"""
LPU Campus Parking - ML inference engine.

Turns a zone's current state (live count + recent history + calendar) into forecasts at +15/+30/+60
minutes with calibrated uncertainty, and interpolates to any arrival time in between. The live
data plumbing (which counts to use, stale cameras, demo feed) lives in models/live.py.
"""

import json
import logging
import joblib
import threading
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Optional
from pathlib import Path

from features import calendar as cal
from features.feature_pipeline import FEATURE_COLS, HORIZONS, horizon_col, zone_code
from models.history import HISTORY, ZoneHistory, naive_ist
from models.uncertainty import clip_forecast, event_flag, free_slots_needed
from utils.config import load_model_config

logger = logging.getLogger(__name__)
_IST = ZoneInfo("Asia/Kolkata")

MODEL_DIR = Path("artifacts/pipelines")
KEYS = {15: "H15", 30: "H30", 60: "H60"}
MAX_HORIZON = 60


# ---------------------------------------------------------------------------
# Model loader (singleton pattern — loaded once at startup)
# ---------------------------------------------------------------------------
_xgb_model = None
_clf_model = None
_int_model = None      # optional CQR intervals
_resid_model = None    # calibrated residual distribution (free-slot probability, intervals)
_profile = None        # normal-day profile
_demand = None         # calendar-only model for arrival times hours / days ahead
_demand_unc = None     # its calibrated uncertainty
_meta: Dict[str, Any] = {}
_model_lock = threading.Lock()
_load_error: Optional[str] = None


def _normalize_timestamp(ts: Optional[datetime]) -> datetime:
    """Consistently timezone-aware Asia/Kolkata timestamp (naive input is assumed to be IST)."""
    if ts is None:
        return datetime.now(_IST)
    if ts.tzinfo is None:
        return ts.replace(tzinfo=_IST)
    return ts.astimezone(_IST)


def _load_models(force: bool = False):
    global _xgb_model, _clf_model, _int_model, _resid_model, _profile, _demand, _demand_unc, _meta, _load_error
    if _xgb_model is not None and not force:
        return True
    with _model_lock:                      # several request threads may hit a cold start together
        if _xgb_model is not None and not force:
            return True
        try:
            need = [MODEL_DIR / n for n in ("xgb_multi_horizon.joblib", "saturation_classifier.joblib",
                                            "residual_dist.joblib", "profile.joblib")]
            missing = [str(p) for p in need if not p.exists()]
            if missing:
                _load_error = f"Model artifacts not found: {missing}. Run: python -m models.train"
                return False
            xgb_m = joblib.load(need[0])
            clf_m = joblib.load(need[1])
            resid_m = joblib.load(need[2])
            prof_m = joblib.load(need[3])
            int_path = MODEL_DIR / "xgb_intervals.joblib"
            int_m = joblib.load(int_path) if int_path.exists() else None
            meta_path = MODEL_DIR / "model_meta.json"
            _meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            dem_p, unc_p = MODEL_DIR / "demand_forecaster.joblib", MODEL_DIR / "demand_uncertainty.joblib"
            _demand = joblib.load(dem_p) if dem_p.exists() else None                    # optional: far arrival times
            _demand_unc = joblib.load(unc_p) if unc_p.exists() else None
            _xgb_model, _clf_model, _resid_model, _profile, _int_model = xgb_m, clf_m, resid_m, prof_m, int_m
            _load_error = None
            logger.info("Forecaster loaded: version=%s data_source=%s", _meta.get("version"), _meta.get("data_source"))
            return True
        except Exception as e:
            _load_error = str(e)
            logger.error(f"Failed to load ML models: {e}")
            return False


def models_ready() -> Dict[str, Any]:
    ok = _load_models()
    return {"loaded": ok, "error": _load_error}


def model_meta() -> Dict[str, Any]:
    """Version, training data source and headline metrics of the loaded model (empty if none)."""
    _load_models()
    return dict(_meta)


def get_profile():
    _load_models()
    return _profile


# ---------------------------------------------------------------------------
# Feature construction from live inputs
# ---------------------------------------------------------------------------

def _build_feature_row(
    zone_id: str,
    current_count: float,
    operational_capacity: int,
    is_exam_day: Optional[bool] = None,
    is_class_transition: Optional[bool] = None,
    timestamp: Optional[datetime] = None,
    lag_counts: Optional[Dict[str, int]] = None,  # {'15m':x, '30m':x, '45m':x, '60m':x}
    history: Optional[ZoneHistory] = None,
    is_campus_event: Optional[bool] = None,
    entries: Optional[List[cal.CalendarEntry]] = None,
):
    """
    Build the feature vector in the exact schema used for training (features.feature_pipeline.FEATURE_COLS).

    Calendar flags (exam, holiday, event, bell window) come from the campus calendar unless an explicit
    override is given (None = automatic). Lag sources, in priority order: explicit ``lag_counts`` >
    rolling ``history`` buffer > current count (flagged as "estimated").
    Returns (DataFrame, lag_source) where lag_source is "observed", "partial" or "estimated".
    """
    ts = naive_ist(_normalize_timestamp(timestamp))
    times = [ts] + [ts + timedelta(minutes=h) for h in HORIZONS]
    fl = cal.compute_flags(times, zone_id, entries)
    exam_now = int(fl["is_exam_day"][0]) if is_exam_day is None else int(is_exam_day)
    bell_now = int(fl["is_class_transition"][0]) if is_class_transition is None else int(is_class_transition)
    event_now = int(fl["is_campus_event"][0]) if is_campus_event is None else int(is_campus_event)

    hour_frac = ts.hour + ts.minute / 60.0
    dow = ts.weekday()  # 0=Mon ... 6=Sun
    occ_ratio = current_count / operational_capacity if operational_capacity > 0 else 0.0

    lags: Dict[int, Optional[float]] = {}
    for m in (15, 30, 45, 60):
        v = None
        if lag_counts and f"{m}m" in lag_counts:
            v = lag_counts[f"{m}m"]
        elif lag_counts and f"lag_{m}m" in lag_counts:
            v = lag_counts[f"lag_{m}m"]
        elif history is not None:
            v = history.lag(zone_id, ts, m)
        lags[m] = v
    found = sum(v is not None for v in lags.values())
    lag_source = "observed" if found == 4 else "partial" if found else "estimated"
    lag_val = {m: (v if v is not None else current_count) for m, v in lags.items()}

    # Rolling stats: training uses the 5-minute samples t-5..t-W (current step
    # excluded, pandas sample std). Use the same window when history exists.
    def _roll(w: int):
        vals = history.window(zone_id, ts, w) if history is not None else []
        if not vals:  # fall back to whatever lags we have
            vals = [lag_val[m] for m in (15, 30, 45, 60) if m <= w] or [current_count]
        arr = np.asarray(vals, dtype=float)
        std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        return float(arr.mean()), std

    rm30, rs30 = _roll(30)
    rm60, rs60 = _roll(60)

    zc = zone_code(zone_id)
    if _profile is not None:
        prof = _profile.lookup([zc] * len(times), times, fl["is_holiday"].to_numpy())
    else:
        prof = np.full(len(times), occ_ratio)
    exam_future = lambda i: exam_now if is_exam_day is not None else int(fl["is_exam_day"][i])   # what-if applies to the whole day

    row = {
        'vehicle_count':        current_count,
        'operational_capacity': operational_capacity,
        'occupancy_ratio':      occ_ratio,
        'zone_code':            zc,
        'hour_sin':             np.sin(2 * np.pi * hour_frac / 24.0),
        'hour_cos':             np.cos(2 * np.pi * hour_frac / 24.0),
        'day_sin':              np.sin(2 * np.pi * dow / 7.0),
        'day_cos':              np.cos(2 * np.pi * dow / 7.0),
        'is_weekend':           int(dow >= 5),
        'is_holiday':           int(fl["is_holiday"][0]),
        'is_exam_day':          exam_now,
        'is_campus_event':      event_now,
        'event_load':           float(fl["event_load"][0]),
        'is_class_transition':  bell_now,
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
        'profile_now':          float(prof[0]),
        'profile_dev':          occ_ratio - float(prof[0]),
    }
    for i, h in enumerate(HORIZONS, start=1):
        row[horizon_col(h, "profile")] = float(prof[i])
        row[horizon_col(h, "event_load")] = float(fl["event_load"][i])
        row[horizon_col(h, "is_exam_day")] = exam_future(i)
        row[horizon_col(h, "is_holiday")] = int(fl["is_holiday"][i])
        row[horizon_col(h, "mins_to_bell")] = float(fl["mins_to_bell"][i])
    return pd.DataFrame([row], columns=FEATURE_COLS), lag_source


# ---------------------------------------------------------------------------
# Batch forecasting at the three trained horizons
# ---------------------------------------------------------------------------

def forecast_batch(inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Forecast a batch of zone states. Each input: zone_id, current_count, capacity, timestamp, and
    optionally history / lag_counts / is_exam_day / is_class_transition / is_campus_event.

    Each result holds, for horizons 15/30/60: the point forecast, an 80% interval, P(free slot at
    arrival) and P(>=90% full), plus the normal-day profile count, lag_source and inference_source.
    """
    _load_models()
    rows, lag_sources = [], []
    for it in inputs:
        X, ls = _build_feature_row(
            it["zone_id"], it["current_count"], it["capacity"], it.get("is_exam_day"),
            it.get("is_class_transition"), it["timestamp"], it.get("lag_counts"), it.get("history"),
            it.get("is_campus_event"))
        rows.append(X)
        lag_sources.append(ls)
    X = pd.concat(rows, ignore_index=True)
    cap = X["operational_capacity"].to_numpy(dtype=float)
    cur = X["vehicle_count"].to_numpy(dtype=float)
    need = np.array([free_slots_needed(c) for c in cap])
    have_model = _xgb_model is not None and _resid_model is not None

    pts, lo, hi, pfree, psat, prof = {}, {}, {}, {}, {}, {}
    for h in HORIZONS:
        k = KEYS[h]
        prof[h] = X[horizon_col(h, "profile")].to_numpy(dtype=float) * cap
        near = (X[horizon_col(h, "mins_to_bell")].to_numpy() <= 15).astype(float)
        ev = event_flag(X, h)
        if have_model:
            p = clip_forecast(_xgb_model.predict_horizon(X, k), cur, cap)
            pts[h] = p
            lo[h], hi[h] = _resid_model.interval(k, p, cap, near, coverage=load_model_config()["uncertainty"]["interval_coverage"],
                                                 event=ev)
            if _int_model is not None:                             # optional feature-aware CQR band replaces it
                lo[h], hi[h] = _int_model.predict(X, k, point=p)
            pfree[h] = _resid_model.p_free(k, p, cap, near, event=ev)
            psat[h] = _resid_model.p_saturated(k, p, cap, near, event=ev)
        else:                                                      # no trained model: normal-day profile, no uncertainty
            pts[h] = np.clip(prof[h], 0.0, cap * 1.30)
            lo[h] = hi[h] = pts[h]
            pfree[h] = (pts[h] <= cap - need).astype(float)
            psat[h] = (pts[h] >= 0.9 * cap).astype(float)
    if have_model:
        psat[15] = np.asarray(_clf_model.predict_proba(X), dtype=float)   # calibrated classifier at +15

    out = []
    for i, it in enumerate(inputs):
        out.append({
            "zone_id": it["zone_id"], "capacity": int(cap[i]), "current_count": float(cur[i]),
            "pts": {h: float(pts[h][i]) for h in HORIZONS}, "low": {h: float(lo[h][i]) for h in HORIZONS},
            "high": {h: float(hi[h][i]) for h in HORIZONS}, "p_free": {h: float(pfree[h][i]) for h in HORIZONS},
            "p_sat": {h: float(psat[h][i]) for h in HORIZONS}, "profile": {h: float(prof[h][i]) for h in HORIZONS},
            "lag_source": lag_sources[i],
            "inference_source": "xgboost_model" if have_model else "profile_fallback",
        })
    return out


def calendar_forecast(zone_ids: List[str], times: List[datetime], caps: List[float]) -> List[Dict[str, float]]:
    """
    Calendar-only forecast for each (zone, time): what the campus calendar (exams, holidays, events) and the time of
    week say, without the live count. Used for arrival times beyond +60 minutes. Falls back to the normal-day profile
    plus declared event load (no uncertainty) when the calendar model is not available.
    """
    _load_models()
    caps_a = np.asarray(caps, dtype=float)
    if _demand is not None:
        ratio, ev = _demand.predict_at(zone_ids, times)
        if _demand_unc is not None:
            s = _demand_unc.summarize(ratio, caps_a, ev, coverage=load_model_config()["uncertainty"]["interval_coverage"])
        else:
            p = ratio * caps_a
            s = {"pred": p, "low": p, "high": p, "p_free": None, "p_sat": None}
    else:
        fl = cal.compute_flags(times, list(zone_ids))
        prof = (_profile.lookup([zone_code(z) for z in zone_ids], times, fl["is_holiday"].to_numpy())
                if _profile is not None else np.full(len(zone_ids), 0.4))
        p = np.clip(prof + fl["event_load"].to_numpy(), 0, 1.15) * caps_a
        s = {"pred": p, "low": p, "high": p, "p_free": None, "p_sat": None}
    out = []
    for i, c in enumerate(caps_a):
        need = free_slots_needed(c)
        pf = float(s["p_free"][i]) if s["p_free"] is not None else float(s["pred"][i] <= c - need)
        ps = float(s["p_sat"][i]) if s["p_sat"] is not None else float(s["pred"][i] >= 0.9 * c)
        out.append({"pred": float(s["pred"][i]), "low": float(s["low"][i]), "high": float(s["high"][i]),
                    "p_free": pf, "p_sat": ps})
    return out


def far_weight(horizon_min: float) -> float:
    """How much of the live forecast's deviation survives at this horizon (1 up to +60 min, then fades)."""
    tau = load_model_config()["live"]["far_decay_minutes"]
    return 1.0 if horizon_min <= MAX_HORIZON else float(np.exp(-(horizon_min - MAX_HORIZON) / tau))


def at_horizon(fc: Dict[str, Any], horizon_min: float, far: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, Any]:
    """
    Forecast for an arbitrary horizon (minutes from the observation the forecast was made on).

    Up to +60: interpolate between "now" (0) and the trained +15/+30/+60 forecasts.
    Beyond +60: the calendar model at the ARRIVAL time, plus whatever the live forecast says is unusual today,
    fading with far_decay_minutes (measured in experiments/far_horizon.py). `far` = {"arrival": calendar forecast
    at the arrival time, "base60": calendar forecast at observation + 60 min}. Without it, the normal-day profile
    is used (less accurate; kept for callers that cannot supply the calendar forecast).
    """
    cap, cur = fc["capacity"], fc["current_count"]
    need = free_slots_needed(cap)
    h = max(0.0, float(horizon_min))
    now_free = 1.0 if cur <= cap - need else 0.0
    xs = [0, 15, 30, 60]
    point = [cur] + [fc["pts"][k] for k in (15, 30, 60)]
    low = [cur] + [fc["low"][k] for k in (15, 30, 60)]
    high = [cur] + [fc["high"][k] for k in (15, 30, 60)]
    pfree = [now_free] + [fc["p_free"][k] for k in (15, 30, 60)]
    psat = [1.0 if cur >= 0.9 * cap else 0.0] + [fc["p_sat"][k] for k in (15, 30, 60)]
    hh = min(h, MAX_HORIZON)
    res = {k: float(np.interp(hh, xs, v)) for k, v in
           (("predicted", point), ("low", low), ("high", high), ("p_free", pfree), ("p_sat", psat))}
    res["basis"] = "live"
    if h > MAX_HORIZON:
        w = far_weight(h)
        if far is None:      # fallback: normal-day profile at t+60 stands in for the calendar model
            p60 = fc["profile"][60]
            grow = 1.0 + (h - MAX_HORIZON) / 60.0      # no calibrated range here: widen the +60 range, never narrow it
            far = {"arrival": {"pred": p60, "low": p60 - (fc["pts"][60] - fc["low"][60]) * grow,
                               "high": p60 + (fc["high"][60] - fc["pts"][60]) * grow,
                               "p_free": float(p60 <= cap - need), "p_sat": float(p60 >= 0.9 * cap)},
                   "base60": {"pred": p60}}
        arr = far["arrival"]
        res["predicted"] = arr["pred"] + (fc["pts"][60] - far["base60"]["pred"]) * w
        lo_w = w * (fc["pts"][60] - fc["low"][60]) + (1 - w) * (arr["pred"] - arr["low"])
        hi_w = w * (fc["high"][60] - fc["pts"][60]) + (1 - w) * (arr["high"] - arr["pred"])
        res["low"] = max(0.0, res["predicted"] - lo_w)
        res["high"] = res["predicted"] + hi_w
        res["p_free"] = w * fc["p_free"][60] + (1 - w) * arr["p_free"]
        res["p_sat"] = w * fc["p_sat"][60] + (1 - w) * arr["p_sat"]
        res["basis"] = "blended" if w >= 0.1 else "calendar"
    res["predicted"] = float(np.clip(res["predicted"], 0.0, cap * 1.30))
    res["low"] = min(res["low"], res["predicted"])
    res["high"] = max(res["high"], res["predicted"])
    res["horizon_min"] = h
    return res


# ---------------------------------------------------------------------------
# Public inference API (explicit current count; used by /api/predict)
# ---------------------------------------------------------------------------

def predict_zone(
    zone_id: str,
    current_count: int,
    operational_capacity: int,
    horizon_minutes: int = 15,
    is_exam_day: Optional[bool] = None,
    is_class_transition: Optional[bool] = None,
    timestamp: Optional[datetime] = None,
    lag_counts: Optional[Dict[str, int]] = None,
    is_campus_event: Optional[bool] = None,
    history: Optional[ZoneHistory] = HISTORY,
    record: bool = True,
) -> Dict[str, Any]:
    """
    Forecast one zone from an explicit current count, at any horizon from 1 to 60 minutes.
    Falls back to the normal-day profile if the trained model is unavailable (disclosed in
    inference_source).
    """
    timestamp = _normalize_timestamp(timestamp)
    fc = forecast_batch([{
        "zone_id": zone_id, "current_count": current_count, "capacity": operational_capacity,
        "timestamp": timestamp, "history": history, "lag_counts": lag_counts, "is_exam_day": is_exam_day,
        "is_class_transition": is_class_transition, "is_campus_event": is_campus_event}])[0]
    r = at_horizon(fc, horizon_minutes)

    is_frozen = False
    if record and history is not None:
        history.record(zone_id, timestamp, current_count)
        is_frozen = history.is_frozen(zone_id)

    cap = operational_capacity
    pred_count = int(round(r["predicted"]))
    low, high = int(np.floor(r["low"])), int(np.ceil(r["high"]))
    occ_rate = round(min(1.0, pred_count / cap), 4) if cap > 0 else 0.0
    status = ("SATURATED" if occ_rate >= 0.90 else "FILLING_FAST" if occ_rate >= 0.75 else "AVAILABLE")
    return {
        "predicted_vehicle_count": pred_count,
        "predicted_low":           low if fc["inference_source"] == "xgboost_model" else None,
        "predicted_high":          high if fc["inference_source"] == "xgboost_model" else None,
        "operational_capacity":    cap,
        "available_spaces":        max(0, cap - pred_count),
        "occupancy_rate":          occ_rate,
        "saturation_probability":  round(r["p_sat"], 4),
        "p_free":                  round(r["p_free"], 4),
        "status":                  status,
        "inference_source":        fc["inference_source"],
        "lag_source":              fc["lag_source"],
        "stream_frozen":           is_frozen,
        "horizon_minutes":         horizon_minutes,
    }


def predict_all_zones(zones_map: Dict[str, Any], is_exam_day: Optional[bool] = None,
                      is_class_transition: Optional[bool] = None, timestamp: Optional[datetime] = None,
                      horizon_minutes: int = 15) -> Dict[str, Dict[str, Any]]:
    """Forecast every zone from the live feed (see models/live.py). Returns zone_id -> prediction dict."""
    from models import live
    return live.predict_all(zones_map, timestamp, horizon_minutes, is_exam_day, is_class_transition)
