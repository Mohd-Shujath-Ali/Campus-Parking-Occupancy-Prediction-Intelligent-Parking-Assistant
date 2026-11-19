"""
Prediction intervals via Conformalized Quantile Regression (CQR).

1. Fit a lower and an upper quantile XGBoost (alpha/2 and 1-alpha/2) on the TRAIN period.
2. On the later VALIDATION period measure how far the true value falls outside the
   quantile band, and take the (1-alpha) empirical quantile q of those scores.
3. Widen the band by q: [lo - q, hi + q]. Under exchangeability this guarantees
   >= (1-alpha) coverage; for time series it is approximate, so the real coverage
   on the untouched TEST period is reported rather than assumed.
"""
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

from models.pipeline import build_pipeline
from utils.config import xgb_params


class PredictionIntervals:
    def __init__(self, alpha: float = 0.2, horizons=("H15", "H30", "H60")):
        self.alpha = alpha
        self.horizons = list(horizons)
        self.lo: Dict[str, object] = {}
        self.hi: Dict[str, object] = {}
        self.q: Dict[str, float] = {}

    def _quantile_model(self, alpha: float):
        # Deliberately compact (not the tuned point-forecast model): the conformal step below
        # fixes the coverage, so deep trees would only bloat the artefact (>100 MB).
        p = xgb_params("xgboost_regressor", tuned=False)
        p.update(objective="reg:quantileerror", quantile_alpha=alpha, n_estimators=300,
                 max_depth=8, learning_rate=0.05, min_child_weight=5)
        p.pop("eval_metric", None)
        return build_pipeline(xgb.XGBRegressor(**p))

    def fit(self, X_train, y_train: Dict[str, pd.Series], X_val, y_val: Dict[str, pd.Series]):
        for h in self.horizons:
            self.lo[h] = self._quantile_model(self.alpha / 2).fit(X_train, y_train[h])
            self.hi[h] = self._quantile_model(1 - self.alpha / 2).fit(X_train, y_train[h])
            lo, hi = self.lo[h].predict(X_val), self.hi[h].predict(X_val)
            scores = np.maximum(lo - y_val[h].values, y_val[h].values - hi)
            n = len(scores)
            level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
            self.q[h] = float(np.quantile(scores, level))
        return self

    def predict(self, X, horizon: str, point: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        lo = self.lo[horizon].predict(X) - self.q[horizon]
        hi = self.hi[horizon].predict(X) + self.q[horizon]
        if point is not None:                      # band always contains the point forecast
            lo, hi = np.minimum(lo, point), np.maximum(hi, point)
        return np.maximum(lo, 0.0), hi

    def coverage(self, X, y: pd.Series, horizon: str) -> Dict[str, float]:
        lo, hi = self.predict(X, horizon)
        return {"coverage": float(((y.values >= lo) & (y.values <= hi)).mean()),
                "mean_width": float((hi - lo).mean())}
