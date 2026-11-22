"""
Model monitoring: log live forecasts, then score them against the real counts that arrive later.

Every camera-based forecast is stored (zone, horizon, predicted value, 80% band, P(free slot)). Once the
count for the target time has been ingested, the forecast can be compared with what actually happened, giving
live MAE, bias, interval coverage and free-slot calibration per zone and horizon, and a drift warning when
live error runs well above the error measured at training time.
"""
from datetime import datetime, timedelta
from typing import Any, Dict, List
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from api.db import get_db
from models.uncertainty import brier, free_slots_needed
from utils.config import load_model_config

_IST = ZoneInfo("Asia/Kolkata")
MIN_SAMPLES_FOR_DRIFT = 50


def log_predictions(rows: List[Dict[str, Any]]) -> None:
    """One row per (zone, forecast time, horizon); re-forecasting the same reading is ignored."""
    with get_db().conn() as c:
        c.executemany(
            "INSERT OR IGNORE INTO prediction_log (zone_id, made_bucket, horizon_min, target_bucket, predicted, low,"
            " high, p_free, capacity, model_version) VALUES (:zone_id, :made_bucket, :horizon_min, :target_bucket,"
            " :predicted, :low, :high, :p_free, :capacity, :model_version)", rows)


def _scored(store, days: int) -> pd.DataFrame:
    now = datetime.now(_IST).replace(tzinfo=None)
    since = now - timedelta(days=days)
    with get_db().conn() as c:
        log = pd.read_sql_query(
            "SELECT * FROM prediction_log WHERE made_bucket >= ? AND target_bucket <= ?",
            c, params=(since.isoformat(), now.isoformat()))
    if log.empty:
        return log
    obs = store.zone_totals(since=since)
    if obs.empty:
        return log.iloc[0:0]
    obs = obs.assign(target_bucket=obs["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S"))[["zone_id", "target_bucket", "vehicle_count"]]
    df = log.merge(obs, on=["zone_id", "target_bucket"], how="inner").rename(columns={"vehicle_count": "actual"})
    return df


def live_accuracy(store, days: int = 7) -> Dict[str, Any]:
    """Live MAE, bias, coverage and free-slot Brier score per horizon and zone, plus a drift verdict."""
    df = _scored(store, days)
    out: Dict[str, Any] = {"window_days": days, "scored_forecasts": int(len(df)), "horizons": {}, "zones": [],
                           "drift": {"status": "no_data", "detail": "No camera-based forecasts have been scored yet."}}
    if df.empty:
        return out
    df["abs_err"] = (df["actual"] - df["predicted"]).abs()
    df["need"] = df["capacity"].map(free_slots_needed)
    df["free_outcome"] = (df["actual"] <= df["capacity"] - df["need"]).astype(float)
    baseline = load_model_config()["live"]["drift_mae_ratio"]
    from models import inference
    train_mae = {k: v.get("MAE") for k, v in inference.model_meta().get("metrics_test", {}).items() if isinstance(v, dict)}

    worst, flagged = 0.0, []
    for h, g in df.groupby("horizon_min"):
        key = f"H{int(h)}"
        row = {"n": int(len(g)), "MAE": round(float(g.abs_err.mean()), 2),
               "bias": round(float((g.predicted - g.actual).mean()), 2),
               "interval_coverage": round(float(((g.actual >= g.low) & (g.actual <= g.high)).mean()), 3),
               "free_slot_brier": round(brier(g.p_free.to_numpy(), g.free_outcome.to_numpy()), 4),
               "training_MAE": train_mae.get(key)}
        if row["training_MAE"]:
            row["mae_vs_training"] = round(row["MAE"] / row["training_MAE"], 2)
            if row["n"] >= MIN_SAMPLES_FOR_DRIFT and row["mae_vs_training"] > baseline:
                flagged.append(f"{key}: live MAE {row['MAE']} is {row['mae_vs_training']}x the training MAE {row['training_MAE']}")
            worst = max(worst, row["mae_vs_training"] if row["n"] >= MIN_SAMPLES_FOR_DRIFT else 0.0)
        out["horizons"][key] = row
    out["zones"] = [{"zone_id": z, "n": int(len(g)), "MAE": round(float(g.abs_err.mean()), 2),
                     "bias": round(float((g.predicted - g.actual).mean()), 2)}
                    for z, g in df[df.horizon_min == 15].groupby("zone_id")]
    enough = any(r["n"] >= MIN_SAMPLES_FOR_DRIFT for r in out["horizons"].values())
    if flagged:
        out["drift"] = {"status": "drift", "detail": "; ".join(flagged) + ". Consider retraining: python -m models.retrain"}
    elif enough:
        out["drift"] = {"status": "ok", "detail": "Live error is in line with the error measured at training time."}
    else:
        out["drift"] = {"status": "collecting", "detail": f"Need at least {MIN_SAMPLES_FOR_DRIFT} scored forecasts per horizon."}
    return out
