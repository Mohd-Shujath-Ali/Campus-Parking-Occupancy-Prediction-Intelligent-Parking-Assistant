"""
Live forecasting: connects real-time detection counts to the forecaster.

For every zone, in order of preference:
  1. camera    - a count ingested within the freshness window. Forecast from that reading, with the time
                 since the reading added to the horizon. Older readings (up to stale_seconds) are still used,
                 marked "stale" and given lower confidence.
  2. simulated - demo feed (models/simulator.py). Only while NO camera has ever reported, so a camera
                 outage can never silently turn into invented numbers. Disabled with PARKING_DEMO_FEED=off.
  3. profile   - once real data has been seen, a zone with no usable reading gets the normal-day profile for
                 this time (plus any declared event load) as its estimate, marked low confidence.

Forecasts for all zones are made in one batch and cached for 30 s; ingesting a count or editing the
calendar invalidates the cache. Camera-based forecasts are logged so their accuracy can be measured when the
real count for the target time arrives (see api/model_monitor.py).
"""
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np

from features import calendar as cal
from models import inference, simulator
from models.history import HISTORY, ZoneHistory, naive_ist
from utils.config import load_model_config

logger = logging.getLogger(__name__)
_IST = ZoneInfo("Asia/Kolkata")
CACHE_TTL_S = 30
LAG_STEPS_MIN = 60

_lock = threading.RLock()
_ever_real = False
_version = 0
_cache: Dict[Any, Tuple[float, Dict[str, "ZoneForecast"]]] = {}
_prediction_logger: Optional[Callable[[List[Dict[str, Any]]], None]] = None


# ---------------------------------------------------------------------------
# State of the live feed
# ---------------------------------------------------------------------------

def bump() -> None:
    """Drop cached forecasts (new count ingested or calendar edited)."""
    global _version
    with _lock:
        _version += 1
        _cache.clear()


def note_real_data() -> None:
    """A real count arrived: from now on the demo feed is never used."""
    global _ever_real
    _ever_real = True
    bump()


def set_prediction_logger(fn: Optional[Callable[[List[Dict[str, Any]]], None]]) -> None:
    global _prediction_logger
    _prediction_logger = fn


def rehydrate(store, minutes: int = 130) -> int:
    """After a restart, reload recent counts from the observation store into the rolling history."""
    global _ever_real
    stats = store.stats()
    _ever_real = _ever_real or stats["rows"] > 0
    if not stats["rows"]:
        return 0
    since = datetime.now(_IST).replace(tzinfo=None) - timedelta(minutes=minutes)
    df = store.zone_totals(since=since)
    for r in df.itertuples():
        HISTORY.record(r.zone_id, r.timestamp.to_pydatetime(), int(r.vehicle_count))
    bump()
    return len(df)


def demo_feed_enabled() -> bool:
    return os.environ.get("PARKING_DEMO_FEED", "auto").lower() != "off" and not _ever_real and simulator.available()


def now_ist() -> datetime:
    return datetime.now(_IST).replace(tzinfo=None)


# ---------------------------------------------------------------------------
# Per-zone state
# ---------------------------------------------------------------------------

@dataclass
class ZoneState:
    zone_id: str
    capacity: int
    count: int
    source: str                 # camera | simulated | profile
    status: str                 # live | stale | demo | estimated
    anchor: datetime            # time the count refers to (naive IST)
    age_s: float                # now - anchor
    history: Optional[ZoneHistory]
    frozen: bool = False


def _floor5(t: datetime) -> datetime:
    return t.replace(minute=t.minute - t.minute % 5, second=0, microsecond=0)


def get_states(now: datetime, zones_map: Dict[str, dict]) -> Dict[str, ZoneState]:
    live_cfg = load_model_config()["live"]
    fresh, stale = live_cfg["fresh_seconds"], live_cfg["stale_seconds"]
    demo = demo_feed_enabled()
    anchor_now = _floor5(now)
    sim_times = [anchor_now - timedelta(minutes=m) for m in range(LAG_STEPS_MIN, -1, -5)] if demo else []
    sim_counts = simulator.counts_at(sim_times, zones_map) if demo else {}
    profile = inference.get_profile()

    states: Dict[str, ZoneState] = {}
    for zid, meta in zones_map.items():
        cap = meta["operational_capacity"]
        rec = HISTORY.latest(zid)
        if rec is not None:
            age = max(0.0, (now - rec[0]).total_seconds())
            if age <= stale:
                states[zid] = ZoneState(zid, cap, int(rec[1]), "camera", "live" if age <= fresh else "stale",
                                        rec[0], age, HISTORY, HISTORY.is_frozen(zid))
                continue
        if demo:
            hist = simulator.history_for(zid, anchor_now, sim_counts[zid][:-1], sim_times[:-1])
            states[zid] = ZoneState(zid, cap, int(sim_counts[zid][-1]), "simulated", "demo", anchor_now,
                                    (now - anchor_now).total_seconds(), hist)
            continue
        # No usable reading and real data has been seen: estimate from the normal day (+ declared event load)
        fl = cal.flags_at(anchor_now, zid)
        ratio = (profile.at(zid, anchor_now, int(fl["is_holiday"])) if profile is not None else 0.4) + fl["event_load"]
        est = int(round(min(1.15, max(0.0, ratio)) * cap))
        hist = ZoneHistory()
        for m in range(LAG_STEPS_MIN, 0, -5):
            t = anchor_now - timedelta(minutes=m)
            f2 = cal.flags_at(t, zid)
            r2 = (profile.at(zid, t, int(f2["is_holiday"])) if profile is not None else 0.4) + f2["event_load"]
            hist.record(zid, t, int(round(min(1.15, max(0.0, r2)) * cap)))
        states[zid] = ZoneState(zid, cap, est, "profile", "estimated", anchor_now,
                                (now - anchor_now).total_seconds(), hist)
    return states


# ---------------------------------------------------------------------------
# Forecasts
# ---------------------------------------------------------------------------

@dataclass
class ZoneForecast:
    state: ZoneState
    fc: Dict[str, Any]          # inference.forecast_batch result


def forecast_all(zones_map: Dict[str, dict], now: Optional[datetime] = None,
                 is_exam_day: Optional[bool] = None, is_class_transition: Optional[bool] = None
                 ) -> Dict[str, ZoneForecast]:
    now = now or now_ist()
    key = (int(now.timestamp() // CACHE_TTL_S), _version, is_exam_day, is_class_transition)
    with _lock:
        hit = _cache.get(key)
        if hit is not None:
            return hit[1]
    states = get_states(now, zones_map)
    ids = list(zones_map)
    fcs = inference.forecast_batch([{
        "zone_id": z, "current_count": states[z].count, "capacity": states[z].capacity,
        "timestamp": states[z].anchor, "history": states[z].history,
        "is_exam_day": is_exam_day, "is_class_transition": is_class_transition} for z in ids])
    out = {z: ZoneForecast(states[z], fc) for z, fc in zip(ids, fcs)}
    with _lock:
        if len(_cache) > 64:
            _cache.clear()
        _cache[key] = (now.timestamp(), out)
    _log_predictions(out)
    return out


def _log_predictions(forecasts: Dict[str, ZoneForecast]) -> None:
    if _prediction_logger is None or not load_model_config()["live"]["log_predictions"]:
        return
    version = inference.model_meta().get("version")
    rows = []
    for z, f in forecasts.items():
        if f.state.source != "camera" or f.fc["inference_source"] != "xgboost_model":
            continue
        for h in (15, 30, 60):
            rows.append({"zone_id": z, "made_bucket": f.state.anchor.isoformat(),
                         "horizon_min": h, "target_bucket": (f.state.anchor + timedelta(minutes=h)).isoformat(),
                         "predicted": f.fc["pts"][h], "low": f.fc["low"][h], "high": f.fc["high"][h],
                         "p_free": f.fc["p_free"][h], "capacity": f.fc["capacity"], "model_version": version})
    if rows:
        try:
            _prediction_logger(rows)
        except Exception as e:                              # logging must never break serving
            logger.error("Prediction log write failed: %s", e)


def confidence_of(f: ZoneForecast, band_width_frac: float) -> str:
    s = f.state
    if s.source == "simulated":
        return "demo"
    if s.source == "profile" or s.frozen:
        return "low"
    if s.status == "stale" or f.fc["lag_source"] != "observed" or band_width_frac > 0.35:
        return "medium"
    return "high"


def describe(f: ZoneForecast, horizon_minutes: int, far: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Everything the API reports about one zone for an arrival `horizon_minutes` from now."""
    s, fc = f.state, f.fc
    cap = s.capacity
    r = inference.at_horizon(fc, horizon_minutes + s.age_s / 60.0, far)
    pred = int(round(r["predicted"]))
    low, high = int(np.floor(r["low"])), int(np.ceil(r["high"]))
    occ = round(min(1.0, pred / cap), 4) if cap > 0 else 0.0
    have_model = fc["inference_source"] == "xgboost_model"
    camera_status = {"live": "live", "stale": "stale", "demo": "demo", "estimated": "offline"}[s.status]
    return {
        "predicted_vehicle_count": pred,
        "predicted_low": low if have_model else None,
        "predicted_high": high if have_model else None,
        "available_spaces": max(0, cap - pred),
        "occupancy_rate": occ,
        "saturation_probability": round(r["p_sat"], 4),
        "p_free": round(r["p_free"], 4),
        "status": "SATURATED" if occ >= 0.90 else "FILLING_FAST" if occ >= 0.75 else "AVAILABLE",
        "inference_source": fc["inference_source"],
        "lag_source": fc["lag_source"],
        "stream_frozen": s.frozen,
        "current_count": s.count,
        "camera_age_seconds": round(s.age_s, 1),
        "camera_status": camera_status,
        "count_source": s.source,
        "confidence": _far_confidence(confidence_of(f, (high - low) / max(cap, 1)), r["basis"]),
        "horizon_minutes": horizon_minutes,
        "forecast_basis": r["basis"],            # live (<= 60 min) | blended | calendar (hours ahead)
    }


def _far_confidence(conf: str, basis: str) -> str:
    """Hours ahead, today's live count barely matters: a fresh camera no longer earns 'high'."""
    return "medium" if basis == "calendar" and conf == "high" else conf


def predict_all(zones_map: Dict[str, dict], timestamp: Optional[datetime] = None, horizon_minutes: int = 15,
                is_exam_day: Optional[bool] = None, is_class_transition: Optional[bool] = None
                ) -> Dict[str, Dict[str, Any]]:
    """zone_id -> prediction dict for an arrival `horizon_minutes` from now (legacy shape plus P(free) etc.)."""
    now = naive_ist(timestamp) if timestamp is not None else now_ist()
    fcs = forecast_all(zones_map, now, is_exam_day, is_class_transition)
    far = _far_forecasts(fcs, now, horizon_minutes)
    return {z: describe(f, horizon_minutes, far.get(z)) for z, f in fcs.items()}


def _far_forecasts(fcs: Dict[str, ZoneForecast], now: datetime, horizon_minutes: int) -> Dict[str, Dict[str, Any]]:
    """Calendar forecasts at the arrival time and at reading + 60 min, for zones whose arrival is beyond +60."""
    need = [z for z, f in fcs.items() if horizon_minutes + f.state.age_s / 60.0 > inference.MAX_HORIZON]
    if not need:
        return {}
    arrival = now + timedelta(minutes=horizon_minutes)
    caps = [fcs[z].state.capacity for z in need]
    arr = inference.calendar_forecast(need, [arrival] * len(need), caps)
    base = inference.calendar_forecast(need, [fcs[z].state.anchor + timedelta(minutes=60) for z in need], caps)
    return {z: {"arrival": a, "base60": b} for z, a, b in zip(need, arr, base)}


def data_mode(zones_map: Dict[str, dict], now: Optional[datetime] = None) -> Dict[str, Any]:
    """Campus-wide summary of where the counts come from (for health checks and the UI banner)."""
    fcs = forecast_all(zones_map, now)
    by_source: Dict[str, int] = {}
    for f in fcs.values():
        by_source[f.state.source] = by_source.get(f.state.source, 0) + 1
    mode = "demo" if by_source.get("simulated") else "live" if by_source.get("camera") else "estimated"
    return {"mode": mode, "zones_by_source": by_source, "ever_received_real_data": _ever_real}
