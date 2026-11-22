"""
Which modelling choices actually help? Fits each variant on the training period and reports MAE (vehicles)
on the untouched holdout test period, overall and per regime.

Variants:
  persistence / seasonal mean / profile        baselines that use no (or almost no) machine learning
  xgb_old_features                             XGBoost, direct target, WITHOUT the new calendar/profile features
  xgb_direct                                   XGBoost, direct target, new features
  xgb_residual                                 XGBoost, target = deviation from the normal-day profile
  rf_residual                                  bagging (Random Forest), residual target
  blend                                        non-negative Ridge on validation over xgb_residual + rf_residual

Run:  python -m experiments.forecast_ablation [--fast]
Writes artifacts/ablation.csv and artifacts/ablation_regimes.csv
"""
import argparse
import time
import warnings

import numpy as np
import pandas as pd

from evaluation.metrics import compute_regression_metrics
from features.feature_pipeline import FEATURE_COLS, HORIZONS, horizon_col
from models.baselines import PersistenceForecaster, ProfileForecaster, SeasonalHistoricalMeanForecaster
from models.data_prep import load_dataset
from models.forecaster import MultiHorizonForecaster, profile_offset

warnings.filterwarnings("ignore")


class ResidualForecaster(MultiHorizonForecaster):
    """Target = deviation from the normal-day profile at arrival time (measured here, not shipped)."""

    def fit(self, X_train, targets_train, sample_weight=None):
        shifted = {h: np.asarray(targets_train[h], dtype=float) - profile_offset(X_train, h) for h in self.horizons}
        super().fit(X_train, shifted, sample_weight)

    def predict_horizon(self, X, horizon_key):
        return super().predict_horizon(X, horizon_key) + profile_offset(X, horizon_key)

NEW_ONLY = ["is_holiday", "event_load", "profile_now", "profile_dev"] + \
           [horizon_col(h, f) for h in HORIZONS for f in ("profile", "event_load", "is_exam_day", "is_holiday", "mins_to_bell")]


def regimes(X: pd.DataFrame) -> pd.Series:
    r = pd.Series("normal weekday", index=X.index)
    r[X["is_weekend"] == 1] = "weekend"
    r[X["is_holiday"] == 1] = "holiday"
    r[X["is_exam_day"] == 1] = "exam day"
    r[(X["event_load"] > 0) | (X["is_campus_event"] == 1)] = "event"
    return r


def main(fast: bool):
    splits, profile, _ = load_dataset()
    tr, va, te = splits["train"], splits["val"], splits["test"]
    print(f"rows: train {len(tr.X):,} val {len(va.X):,} test {len(te.X):,}")
    xp = {"n_estimators": 150, "max_depth": 8} if fast else {}
    rp = {"n_estimators": 60} if fast else {}

    def strip_new(X):                       # what the previous model could see
        X = X.copy()
        X[NEW_ONLY] = 0.0
        return X

    preds, val_preds, timing = {}, {}, {}

    def run(name, model_type, residual, params, strip=False):
        t0 = time.time()
        cls = ResidualForecaster if residual else MultiHorizonForecaster
        f = cls(model_type=model_type, params=params)
        prep = strip_new if strip else (lambda X: X)
        f.fit(prep(tr.X), tr.y)
        preds[name], val_preds[name] = f.predict(prep(te.X)), f.predict(prep(va.X))
        timing[name] = round(time.time() - t0, 1)
        print(f"  fitted {name:<18} {timing[name]:>6}s", flush=True)

    # baselines
    preds["persistence"] = {k: PersistenceForecaster().predict(te.X) for k in ("H15", "H30", "H60")}
    preds["profile"] = {k: ProfileForecaster(k).predict(te.X) for k in ("H15", "H30", "H60")}
    seas = SeasonalHistoricalMeanForecaster().fit(tr.X, tr.y["H15"])
    preds["seasonal_mean"] = {k: seas.predict(te.X) for k in ("H15", "H30", "H60")}

    run("xgb_old_features", "xgboost", False, xp, strip=True)
    run("xgb_direct", "xgboost", False, xp)
    run("xgb_residual", "xgboost", True, xp)
    run("rf_residual", "random_forest", True, rp)

    # chronological blend: weights fitted on validation, applied to test
    from sklearn.linear_model import Ridge
    blend = {}
    for k in ("H15", "H30", "H60"):
        Zv = np.column_stack([val_preds[m][k] for m in ("xgb_residual", "rf_residual")])
        meta = Ridge(alpha=1.0, positive=True).fit(Zv, va.y[k])
        blend[k] = meta.predict(np.column_stack([preds[m][k] for m in ("xgb_residual", "rf_residual")]))
        print(f"  blend weights {k}: xgb={meta.coef_[0]:.2f} rf={meta.coef_[1]:.2f}")
    preds["blend"] = blend

    reg = regimes(te.X)
    rows, reg_rows = [], []
    base = {k: np.abs(te.y[k].to_numpy() - preds["persistence"][k]).mean() for k in ("H15", "H30", "H60")}
    for name, p in preds.items():
        for k in ("H15", "H30", "H60"):
            y = te.y[k].to_numpy()
            m = compute_regression_metrics(y, np.clip(p[k], 0, None))
            rows.append({"model": name, "horizon": k, "MAE": m["MAE"], "RMSE": m["RMSE"],
                         "skill_vs_persistence_pct": round(100 * (1 - m["MAE"] / base[k]), 1)})
            for r in reg.unique():
                msk = (reg == r).to_numpy()
                reg_rows.append({"model": name, "horizon": k, "regime": r, "n": int(msk.sum()),
                                 "MAE": round(float(np.abs(y[msk] - np.clip(p[k][msk], 0, None)).mean()), 3)})
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\nOverall MAE on holdout test (vehicles; lower is better), skill = % better than persistence")
    print(res.pivot(index="model", columns="horizon", values="MAE").round(2).assign(
        skill_H15=res[res.horizon == "H15"].set_index("model").skill_vs_persistence_pct,
        skill_H60=res[res.horizon == "H60"].set_index("model").skill_vs_persistence_pct).to_string())
    rg = pd.DataFrame(reg_rows)
    print("\nMAE at +60 min by regime")
    print(rg[rg.horizon == "H60"].pivot(index="model", columns="regime", values="MAE").round(2).to_string())
    print("\nMAE at +15 min by regime")
    print(rg[rg.horizon == "H15"].pivot(index="model", columns="regime", values="MAE").round(2).to_string())
    res.to_csv("artifacts/ablation.csv", index=False)
    rg.to_csv("artifacts/ablation_regimes.csv", index=False)
    print("\nfit seconds:", timing)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="smaller models for a quick look")
    main(ap.parse_args().fast)
