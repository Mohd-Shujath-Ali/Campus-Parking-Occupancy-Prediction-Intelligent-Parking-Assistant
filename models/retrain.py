"""
Retrain on collected camera counts, and only keep the new model if it is better (champion / challenger).

Until real counts exist, models are trained on SYNTHETIC data (data_pipeline/generator.py). Once cameras have
been reporting, run this (nightly, or whenever the Model Health page shows drift):

    python -m models.retrain              # train a challenger; promote it only if it wins
    python -m models.retrain --dry-run    # report what would happen, change nothing

How it works
  1. Export the real counts from the observation store, with exam / holiday / event flags from the campus calendar.
  2. Real data goes AFTER the synthetic block on the timeline (shifted by whole weeks so weekdays still line up) and
     counts `real_weight` times as much while there are fewer than `full_real_days` of it. With enough real data the
     synthetic block is dropped entirely.
  3. The last 7 days of real data are the TEST week, the 7 days before are VALIDATION (calibrates uncertainty), everything
     earlier trains the challenger. The current model (champion) is scored on the same test week.
  4. The challenger replaces the champion only if its error is at least `min_gain` lower at +15 and +60 minutes and its
     80% intervals still cover 72-88% of outcomes. The old model is archived under artifacts/archive/<version>/.
"""
import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from models.data_prep import build_splits
from models.trainer import (MODEL_DIR, TrainedModels, build_meta, evaluate_core, fit_core, load_artifacts,
                            save_artifacts)

MIN_REAL_DAYS = 14          # below this there is not enough real data to train and judge a model
FULL_REAL_DAYS = 56         # at or above this, synthetic data is dropped
REAL_WEIGHT = 5.0           # a real row counts this much while mixed with synthetic data
HOLDOUT_DAYS = 7            # test week; the same length again is used for validation
MIN_GAIN = 0.02             # challenger must be at least 2% better (MAE) at +15 and +60
COVERAGE_BOUNDS = (0.72, 0.88)
MIN_ZONES = 6


def _real_frame(store, entries=None) -> pd.DataFrame:
    from data_collection.store import export_training_frame
    return export_training_frame(store, None, entries=entries)


def assemble(real: pd.DataFrame, synthetic: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """Real + (optionally) synthetic data on one timeline, split into train / validation / test-week frames."""
    real = real.copy()
    real["timestamp"] = pd.to_datetime(real["timestamp"])
    days = real["timestamp"].dt.normalize().nunique()
    span_days = (real["timestamp"].max() - real["timestamp"].min()).days + 1
    use_synth = synthetic is not None and span_days < FULL_REAL_DAYS
    real["sample_weight"] = REAL_WEIGHT if use_synth else 1.0
    frames = [real]
    if use_synth:
        s = synthetic.copy()
        s["timestamp"] = pd.to_datetime(s["timestamp"])
        weeks = int(np.ceil((s["timestamp"].max() - real["timestamp"].min()).total_seconds() / (7 * 86400))) + 1
        s["timestamp"] = s["timestamp"] - pd.Timedelta(days=7 * weeks)      # whole weeks: weekdays stay aligned
        s["sample_weight"] = 1.0
        frames.insert(0, s)
    df = pd.concat(frames, ignore_index=True).sort_values(["timestamp", "zone_id"]).reset_index(drop=True)
    end = real["timestamp"].max()
    test_start = end - pd.Timedelta(days=HOLDOUT_DAYS) + pd.Timedelta(minutes=5)
    val_start = test_start - pd.Timedelta(days=HOLDOUT_DAYS)
    gap = pd.Timedelta(hours=1)                                              # buffer so lags/targets never straddle splits
    return {"df": df, "train": df[df.timestamp < val_start - gap], "val": df[(df.timestamp >= val_start) & (df.timestamp < test_start - gap)],
            "test": df[df.timestamp >= test_start], "mode": "mixed" if use_synth else "live",
            "real_days": int(days), "real_span_days": span_days, "real_rows": int(len(real))}


def retrain(store, out_dir: Path = MODEL_DIR, dry_run: bool = False, fast: bool = False,
            synthetic_csv: Optional[str] = "data/processed/lpu_timeseries.csv", entries=None,
            min_real_days: int = MIN_REAL_DAYS, log=print, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Returns {"decision": insufficient_data | kept_champion | promoted | would_promote, ...}."""
    real = _real_frame(store, entries)
    days = real["timestamp"].dt.normalize().nunique() if len(real) else 0
    zones = real["zone_id"].nunique() if len(real) else 0
    log(f"Real observations: {len(real):,} zone-buckets over {days} days, {zones} zones")
    if days < min_real_days or zones < MIN_ZONES:
        msg = (f"Not enough real data to train: need at least {min_real_days} days from {MIN_ZONES}+ zones "
               f"(have {days} days, {zones} zones). Keeping the current model.")
        log(msg)
        return {"decision": "insufficient_data", "message": msg, "real_days": int(days), "zones": int(zones)}

    synthetic = pd.read_csv(synthetic_csv) if synthetic_csv and Path(synthetic_csv).exists() else None
    a = assemble(real, synthetic)
    log(f"Mode: {a['mode']} | train {len(a['train']):,} val {len(a['val']):,} test-week {len(a['test']):,} rows")
    params = params or ({"n_estimators": 80, "max_depth": 8} if fast else None)
    challenger, _ = fit_core(a["train"], a["val"], params=params, log=log)

    test_x = lambda m: build_splits(a["train"], a["val"], a["test"], profile=m.profile)[0]["test"]
    champ = None
    try:
        champ = load_artifacts(out_dir)
    except Exception as e:
        log(f"No usable current model ({e}); the challenger will be promoted.")
    ev_ch = evaluate_core(challenger, test_x(challenger))
    ev_cm = evaluate_core(champ, test_x(champ)) if champ is not None else None

    gains = {k: (round(1 - ev_ch["horizons"][k]["MAE"] / ev_cm["horizons"][k]["MAE"], 3) if ev_cm else None) for k in ("H15", "H30", "H60")}
    cov = {k: ev_ch["horizons"][k]["interval_coverage"] for k in ev_ch["horizons"]}
    cov_ok = all(COVERAGE_BOUNDS[0] <= c <= COVERAGE_BOUNDS[1] for c in cov.values())
    wins = ev_cm is None or (gains["H15"] >= MIN_GAIN and gains["H60"] >= MIN_GAIN)
    promote = bool(wins and cov_ok)
    reason = ("challenger beats the current model on the latest real week" if promote else
              "challenger is not clearly better" if not wins else "challenger's intervals are miscalibrated")
    for k in ("H15", "H30", "H60"):
        log(f"  {k}: challenger MAE {ev_ch['horizons'][k]['MAE']}" + (f" vs current {ev_cm['horizons'][k]['MAE']} ({gains[k]:+.1%})" if ev_cm else "")
            + f" | coverage {cov[k]}")
    result: Dict[str, Any] = {"decision": "kept_champion", "reason": reason, "mode": a["mode"], "real_days": a["real_days"],
                              "gains": gains, "coverage": cov, "challenger": ev_ch["horizons"],
                              "champion": ev_cm["horizons"] if ev_cm else None}
    if promote and dry_run:
        result["decision"] = "would_promote"
    elif promote:
        meta = build_meta(challenger, a["df"], "live" if a["mode"] == "live" else "mixed", {**ev_ch["horizons"]},
                          notes=f"Retrained {datetime.now(timezone.utc):%Y-%m-%d} on {a['real_days']} days of real counts "
                                f"({a['mode']}); promoted because: {reason}.")
        if champ is not None:
            old = json.loads((out_dir / "model_meta.json").read_text()).get("version", "previous") if (out_dir / "model_meta.json").exists() else "previous"
            arch = out_dir.parent / "archive" / old
            arch.mkdir(parents=True, exist_ok=True)
            for f in out_dir.glob("*.joblib"):
                shutil.copy2(f, arch / f.name)
            if (out_dir / "model_meta.json").exists():
                shutil.copy2(out_dir / "model_meta.json", arch / "model_meta.json")
        save_artifacts(challenger, meta, out_dir)
        result.update(decision="promoted", version=meta["version"])
        log(f"Promoted {meta['version']} (archived the previous model).")
    else:
        log(f"Kept the current model: {reason}.")
    if not dry_run:
        reg = out_dir.parent / "registry.json"
        hist = json.loads(reg.read_text()) if reg.exists() else []
        hist.append({"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **{k: result.get(k) for k in
                     ("decision", "reason", "mode", "real_days", "gains", "coverage", "version")}})
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        reg.write_text(json.dumps(hist, indent=2, default=str))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Retrain on real camera counts; promote only if better")
    ap.add_argument("--dry-run", action="store_true", help="decide but change nothing")
    ap.add_argument("--fast", action="store_true", help="smaller models (development)")
    ap.add_argument("--db", default=None, help="observation database (default: PARKING_DB)")
    a = ap.parse_args()
    from api.calendar_store import list_entries
    from data_collection.store import DEFAULT_DB, ObservationStore
    out = retrain(ObservationStore(a.db or DEFAULT_DB), dry_run=a.dry_run, fast=a.fast, entries=list_entries())
    sys.exit(0 if out["decision"] != "insufficient_data" else 2)
