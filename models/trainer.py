"""
Core training and evaluation, shared by models/train.py (full benchmark run) and models/retrain.py
(champion/challenger retraining on collected data).
"""
import hashlib
import json
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost

from features.feature_pipeline import FEATURE_COLS, HORIZONS, horizon_col
from features.profile import ProfileModel
from models.classifier import SaturationClassifier
from models.data_prep import Split, build_splits
from models.forecaster import MultiHorizonForecaster
from models.intervals import PredictionIntervals
from models.uncertainty import (ResidualDistribution, brier, clip_forecast, event_flag, free_slots_needed,
                                reliability_table)
from utils.config import load_model_config

KEYS = {"H15": 15, "H30": 30, "H60": 60}
MODEL_DIR = Path("artifacts/pipelines")


@dataclass
class TrainedModels:
    forecaster: MultiHorizonForecaster
    classifier: SaturationClassifier
    resid: ResidualDistribution
    profile: ProfileModel
    intervals: Optional[PredictionIntervals] = None


def predict_clipped(forecaster: MultiHorizonForecaster, X: pd.DataFrame) -> Dict[str, np.ndarray]:
    cap, cur = X["operational_capacity"].to_numpy(dtype=float), X["vehicle_count"].to_numpy(dtype=float)
    return {k: clip_forecast(v, cur, cap) for k, v in forecaster.predict(X).items()}


def fit_core(df_train: pd.DataFrame, df_val: pd.DataFrame, params: Optional[Dict[str, Any]] = None,
             cqr: bool = False, log=print, profile: Optional[ProfileModel] = None):
    """
    Fit the production models: direct XGBoost forecasters (+15/+30/+60), calibrated saturation classifier,
    residual distribution (calibrated on the validation period) and optionally CQR intervals.
    The normal-day profile is fitted on the training period only. Returns (TrainedModels, splits).
    """
    splits, profile = build_splits(df_train, df_val, None, profile=profile)
    tr, va = splits["train"], splits["val"]
    weights = tr.meta["sample_weight"].to_numpy() if "sample_weight" in tr.meta else None
    log(f"  train {len(tr.X):,} rows, validation {len(va.X):,} rows")

    forecaster = MultiHorizonForecaster("xgboost", params=params)
    forecaster.fit(tr.X, tr.y, sample_weight=weights)
    pred_val = predict_clipped(forecaster, va.X)

    resid = ResidualDistribution().fit(va.X, va.y, pred_val)
    clf = SaturationClassifier().fit(tr.X, tr.y["SAT15"], va.X, va.y["SAT15"])
    intervals = PredictionIntervals(alpha=1 - load_model_config()["uncertainty"]["interval_coverage"]).fit(
        tr.X, tr.y, va.X, va.y) if cqr else None
    return TrainedModels(forecaster, clf, resid, profile, intervals), splits


def evaluate_core(m: TrainedModels, split: Split) -> Dict[str, Any]:
    """MAE / skill vs persistence / interval coverage / free-slot calibration on one split."""
    X = split.X
    cap, cur = X["operational_capacity"].to_numpy(dtype=float), X["vehicle_count"].to_numpy(dtype=float)
    need = np.array([free_slots_needed(c) for c in np.unique(cap)])
    need_of = dict(zip(np.unique(cap), need))
    need_row = np.array([need_of[c] for c in cap])
    preds = predict_clipped(m.forecaster, X)
    cov_target = load_model_config()["uncertainty"]["interval_coverage"]
    out: Dict[str, Any] = {"horizons": {}, "calibration": {}}
    for k, h in KEYS.items():
        y = split.y[k].to_numpy(dtype=float)
        p = preds[k]
        near = (X[horizon_col(h, "mins_to_bell")].to_numpy() <= 15).astype(float)
        ev = event_flag(X, h)
        mae, mae_pers = float(np.abs(y - p).mean()), float(np.abs(y - cur).mean())
        if m.intervals is not None:
            lo, hi = m.intervals.predict(X, k, point=p)
        else:
            lo, hi = m.resid.interval(k, p, cap, near, coverage=cov_target, event=ev)
        pf = m.resid.p_free(k, p, cap, near, event=ev)
        cov_event = float(((y >= lo) & (y <= hi))[ev == 1].mean()) if (ev == 1).any() else None
        outcome = (y <= cap - need_row).astype(float)
        out["horizons"][k] = {
            "MAE": round(mae, 3), "RMSE": round(float(np.sqrt(((y - p) ** 2).mean())), 3),
            "persistence_MAE": round(mae_pers, 3), "skill_vs_persistence_pct": round(100 * (1 - mae / mae_pers), 1),
            "interval_coverage": round(float(((y >= lo) & (y <= hi)).mean()), 3), "interval_target": cov_target,
            "interval_coverage_events": round(cov_event, 3) if cov_event is not None else None,
            "interval_mean_width": round(float((hi - lo).mean()), 2),
            "free_slot_brier": round(brier(pf, outcome), 4),
            "free_slot_brier_climatology": round(brier(np.full_like(pf, outcome.mean()), outcome), 4),
            "free_slot_base_rate": round(float(outcome.mean()), 3),
        }
        out["calibration"][k] = reliability_table(pf, outcome)
    return out


def regime_of(X: pd.DataFrame) -> pd.Series:
    r = pd.Series("normal weekday", index=X.index)
    r[X["is_weekend"] == 1] = "weekend"
    r[X["is_holiday"] == 1] = "holiday"
    r[X["is_exam_day"] == 1] = "exam day"
    r[(X["event_load"] > 0) | (X["is_campus_event"] == 1)] = "event"
    return r


def regime_table(m: TrainedModels, split: Split) -> pd.DataFrame:
    """MAE per regime for the model vs persistence."""
    X = split.X
    preds = predict_clipped(m.forecaster, X)
    cur = X["vehicle_count"].to_numpy(dtype=float)
    reg = regime_of(X)
    rows = []
    for k in KEYS:
        y = split.y[k].to_numpy(dtype=float)
        for r in reg.unique():
            msk = (reg == r).to_numpy()
            mae, pers = float(np.abs(y[msk] - preds[k][msk]).mean()), float(np.abs(y[msk] - cur[msk]).mean())
            rows.append({"horizon": k, "regime": r, "rows": int(msk.sum()), "MAE": round(mae, 3),
                         "persistence_MAE": round(pers, 3), "skill_vs_persistence_pct": round(100 * (1 - mae / pers), 1)})
    return pd.DataFrame(rows)


def fit_demand(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: Optional[pd.DataFrame] = None, log=print):
    """
    Calendar-only demand model (arrival times hours / days ahead) and its uncertainty.
    The uncertainty is calibrated on validation residuals of a TRAIN-only fit; the deployed model is refitted on
    train + validation (slightly more data, so the calibrated ranges are, if anything, conservative).
    Returns (deployed model, CalendarResidualDistribution, test metrics or None).
    """
    from models.demand import DemandForecaster, frame_from_raw
    from models.uncertainty import CalendarResidualDistribution
    dem = DemandForecaster().fit(df_train)

    def parts(df):
        cap = df["operational_capacity"].to_numpy(float)
        return dem.predict_ratio(frame_from_raw(df)), df["vehicle_count"].to_numpy(float) / cap, \
            (df["event_load"].to_numpy() > 0).astype(int), cap
    pv, av, ev, _ = parts(df_val)
    unc = CalendarResidualDistribution().fit(pv, av, ev)
    metrics = None
    if df_test is not None:
        pt, at, et, cap = parts(df_test)
        s = unc.summarize(pt, cap, et)
        actual = at * cap
        need = np.array([free_slots_needed(c) for c in cap])
        outcome = (actual <= cap - need).astype(float)
        metrics = {"MAE": round(float(np.abs(actual - s["pred"]).mean()), 3),
                   "interval_coverage": round(float(((actual >= s["low"]) & (actual <= s["high"])).mean()), 3),
                   "interval_coverage_events": round(float(((actual >= s["low"]) & (actual <= s["high"]))[et == 1].mean()), 3) if et.any() else None,
                   "free_slot_brier": round(brier(s["p_free"], outcome), 4),
                   "free_slot_brier_climatology": round(brier(np.full_like(outcome, outcome.mean()), outcome), 4)}
        log(f"  calendar model (hours/days ahead): MAE {metrics['MAE']} | 80% coverage {metrics['interval_coverage']} "
            f"(events {metrics['interval_coverage_events']}) | P(free) Brier {metrics['free_slot_brier']} vs {metrics['free_slot_brier_climatology']}")
    deployed = DemandForecaster().fit(pd.concat([df_train, df_val]))
    return deployed, unc, metrics


# ---------------------------------------------------------------------------
# Persistence of artifacts
# ---------------------------------------------------------------------------

def data_fingerprint(df: pd.DataFrame) -> str:
    h = hashlib.md5()
    h.update(str(len(df)).encode())
    h.update(str(df["timestamp"].min()).encode() + str(df["timestamp"].max()).encode())
    h.update(str(df["vehicle_count"].sum()).encode())
    return h.hexdigest()[:8]


def build_meta(m: TrainedModels, df_all: pd.DataFrame, data_source: str, metrics: Dict[str, Any],
               notes: str = "") -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    fp = data_fingerprint(df_all)
    return {
        "version": f"{now:%Y%m%d-%H%M}-{fp}",
        "trained_at": now.isoformat(timespec="seconds"),
        "data_source": data_source,                     # synthetic | live | mixed
        "data_from": str(pd.to_datetime(df_all["timestamp"]).min()),
        "data_to": str(pd.to_datetime(df_all["timestamp"]).max()),
        "rows": int(len(df_all)), "zones": int(df_all["zone_id"].nunique()),
        "features": len(FEATURE_COLS), "horizons_min": list(HORIZONS),
        "intervals": "cqr" if m.intervals is not None else "residual_distribution",
        "metrics_test": metrics,
        "libraries": {"python": platform.python_version(), "xgboost": xgboost.__version__,
                      "scikit_learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
        "notes": notes,
    }


def save_artifacts(m: TrainedModels, meta: Dict[str, Any], out_dir: Path = MODEL_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(m.forecaster, out_dir / "xgb_multi_horizon.joblib")
    joblib.dump(m.classifier, out_dir / "saturation_classifier.joblib")
    joblib.dump(m.resid, out_dir / "residual_dist.joblib")
    joblib.dump(m.profile, out_dir / "profile.joblib")
    iv = out_dir / "xgb_intervals.joblib"
    if m.intervals is not None:
        joblib.dump(m.intervals, iv)
    else:
        iv.unlink(missing_ok=True)       # a stale band from an earlier run must not override the new calibration
    (out_dir / "model_meta.json").write_text(json.dumps(meta, indent=2))
    # Portable boosters (survive an XGBoost upgrade): the XGBoost step of each pipeline
    booster_dir = out_dir / "boosters"
    booster_dir.mkdir(exist_ok=True)
    for horizon, model in m.forecaster.models.items():
        model.named_steps["model"].get_booster().save_model(str(booster_dir / f"xgb_{horizon}.ubj"))
    m.classifier.base.named_steps["model"].get_booster().save_model(str(booster_dir / "saturation_clf.ubj"))


def load_artifacts(out_dir: Path = MODEL_DIR) -> TrainedModels:
    iv = out_dir / "xgb_intervals.joblib"
    return TrainedModels(joblib.load(out_dir / "xgb_multi_horizon.joblib"), joblib.load(out_dir / "saturation_classifier.joblib"),
                         joblib.load(out_dir / "residual_dist.joblib"), joblib.load(out_dir / "profile.joblib"),
                         joblib.load(iv) if iv.exists() else None)
