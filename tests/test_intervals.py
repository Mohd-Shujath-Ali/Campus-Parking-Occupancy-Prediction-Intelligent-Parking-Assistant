import numpy as np
import pandas as pd

from features.feature_pipeline import FEATURE_COLS, zone_order
from models.intervals import PredictionIntervals


def _data(n, seed):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({c: rng.random(n) for c in FEATURE_COLS})
    X["zone_code"] = rng.integers(0, len(zone_order()), n)
    y = pd.Series(100 * X["lag_15m"] + rng.normal(0, 5, n))
    return X, {"H15": y}


def test_conformal_coverage_close_to_nominal_on_fresh_data():
    (Xt, yt), (Xv, yv), (Xs, ys) = _data(6000, 1), _data(3000, 2), _data(3000, 3)
    iv = PredictionIntervals(alpha=0.2, horizons=["H15"]).fit(Xt, yt, Xv, yv)
    cov = iv.coverage(Xs, ys["H15"], "H15")
    assert 0.75 <= cov["coverage"] <= 0.88
    lo, hi = iv.predict(Xs, "H15", point=np.full(len(Xs), 50.0))
    assert (lo <= 50).all() and (hi >= 50).all() and (lo >= 0).all()
