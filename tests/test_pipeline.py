import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from features.feature_pipeline import FEATURE_COLS, zone_order
from models.pipeline import build_pipeline, transformed_feature_names


def _X(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({c: rng.random(n) for c in FEATURE_COLS})
    X["zone_code"] = rng.integers(0, len(zone_order()), n)
    return X, X["vehicle_count"] * 10


def test_pipeline_fits_on_raw_features_and_roundtrips(tmp_path):
    X, y = _X()
    pipe = build_pipeline(xgb.XGBRegressor(n_estimators=10)).fit(X, y)
    path = tmp_path / "p.joblib"
    joblib.dump(pipe, path)
    assert np.allclose(joblib.load(path).predict(X), pipe.predict(X))


def test_zone_is_one_hot_encoded_and_names_are_readable():
    X, y = _X()
    pipe = build_pipeline(xgb.XGBRegressor(n_estimators=5)).fit(X, y)
    names = transformed_feature_names(pipe)
    assert sum(n.startswith("zone=") for n in names) == len(zone_order())
    assert "lag_15m" in names and "zone_code" not in names


def test_unseen_zone_code_does_not_crash():
    X, y = _X()
    pipe = build_pipeline(xgb.XGBRegressor(n_estimators=5)).fit(X, y)
    X2 = X.head(3).copy(); X2["zone_code"] = -1  # unknown zone at serving time
    assert len(pipe.predict(X2)) == 3


def test_saturation_classifier_is_calibrated_on_validation():
    from models.classifier import SaturationClassifier
    rng = np.random.default_rng(1)
    def mk(n):
        X, _ = _X(n, seed=int(rng.integers(1e6)))
        y = pd.Series((X["occupancy_ratio"] + rng.normal(0, .15, n) > .9).astype(int))
        return X, y
    (Xt, yt), (Xv, yv), (Xs, ys) = mk(4000), mk(2000), mk(2000)
    raw = SaturationClassifier(n_estimators=40).fit(Xt, yt)
    cal = SaturationClassifier(n_estimators=40).fit(Xt, yt, Xv, yv)
    # scale_pos_weight inflates raw probabilities; calibration brings the mean back to the base rate
    gap_raw = abs(raw.predict_proba(Xs).mean() - ys.mean())
    gap_cal = abs(cal.predict_proba(Xs).mean() - ys.mean())
    assert gap_cal < gap_raw
