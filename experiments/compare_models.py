"""
Model comparison (rubric 3.5): baselines vs linear vs bagging vs boosting vs stacking.

All models share the same chronological split and the same sklearn
preprocessing Pipeline. Metrics are reported on validation and hold-out test.

Run:  python -m experiments.compare_models
"""
import time
import warnings

import lightgbm as lgb
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from evaluation.metrics import compute_inflection_metrics, compute_regression_metrics
from models.baselines import PersistenceForecaster, SeasonalHistoricalMeanForecaster
from models.data_prep import load_splits
from models.ensemble import BlendStackingRegressor
from models.pipeline import build_pipeline
from utils.config import xgb_params

warnings.filterwarnings("ignore")
HORIZONS = {"H15": "+15 min", "H30": "+30 min", "H60": "+60 min"}


def estimators():
    """Candidate models; hyperparameters from model_config.yaml (+ tuned XGBoost)."""
    return {
        "Ridge": build_pipeline(Ridge(**xgb_params("ridge"))),
        "Random Forest": build_pipeline(RandomForestRegressor(**xgb_params("random_forest"))),
        "XGBoost": build_pipeline(xgb.XGBRegressor(**xgb_params("xgboost_regressor"))),
        "LightGBM": build_pipeline(lgb.LGBMRegressor(
            n_estimators=200, learning_rate=0.04, max_depth=6, subsample=0.85,
            subsample_freq=1, colsample_bytree=0.8, random_state=42, verbose=-1)),
    }


def score(y, pred, split):
    m = compute_regression_metrics(y.y_true.values, pred)
    infl = compute_inflection_metrics(y.y_true.values, pred, split.X["is_class_transition"].values)
    return {**m, "Inflection_MAE": infl["MAE_Rush_Inflection"], "Steady_MAE": infl["MAE_Steady_State"]}


class _Y:  # tiny holder so score() gets the aligned truth
    def __init__(self, y): self.y_true = y


def main():
    sp = load_splits()
    tr, va, te = sp["train"], sp["val"], sp["test"]
    print(f"rows  train={len(tr.X)}  val={len(va.X)}  test={len(te.X)}  features={tr.X.shape[1]}")
    rows = []

    def add(name, hz, split_name, split, pred, secs):
        r = score(_Y(split.y[hz]), pred, split)
        rows.append({"Model": name, "Horizon": HORIZONS[hz], "Split": split_name, **r,
                     "Fit_seconds": round(secs, 1)})

    for hz in HORIZONS:
        print(f"\n=== {HORIZONS[hz]} ===")
        # Baselines
        for name, mdl in (("Persistence", PersistenceForecaster()),
                          ("Seasonal mean (zone x weekday x hour)", SeasonalHistoricalMeanForecaster())):
            mdl.fit(tr.X, tr.y[hz])
            for sn, s in (("val", va), ("test", te)):
                add(name, hz, sn, s, mdl.predict(s.X), 0.0)
        # Learned models
        fitted = {}
        for name, est in estimators().items():
            t0 = time.time(); est.fit(tr.X, tr.y[hz]); secs = time.time() - t0
            fitted[name] = est
            for sn, s in (("val", va), ("test", te)):
                add(name, hz, sn, s, est.predict(s.X), secs)
            print(f"  {name:14s} test MAE={rows[-1]['MAE']:.3f}  ({secs:.1f}s)")
        # Stacking (Random Forest + XGBoost + LightGBM -> non-negative Ridge on validation)
        t0 = time.time()
        stack = BlendStackingRegressor(
            {k: v for k, v in estimators().items() if k != "Ridge"}
        ).fit(tr.X, tr.y[hz], va.X, va.y[hz])
        secs = time.time() - t0
        # val score of a stack is optimistic (meta-learner saw val): report test only
        add("Stacking (RF+XGB+LGBM)", hz, "test", te, stack.predict(te.X), secs)
        print(f"  {'Stacking':14s} test MAE={rows[-1]['MAE']:.3f}  weights={ {k: float(v) for k, v in stack.weights_.items()} }")

    out = pd.DataFrame(rows)
    out.to_csv("artifacts/model_comparison_full.csv", index=False)
    t = out[out.Split == "test"].pivot_table(index="Model", columns="Horizon", values="MAE")
    print("\nTEST MAE (cars) by model and horizon:\n", t.round(3).to_string())
    return out


if __name__ == "__main__":
    main()
