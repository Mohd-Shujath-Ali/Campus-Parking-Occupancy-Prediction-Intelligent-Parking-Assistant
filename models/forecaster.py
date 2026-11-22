import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from features.feature_pipeline import horizon_col
from utils.config import xgb_params
from models.pipeline import build_pipeline

HORIZON_MINUTES = {"H15": 15, "H30": 30, "H60": 60}


def profile_offset(X: pd.DataFrame, horizon_key: str) -> np.ndarray:
    """Expected vehicle count at arrival time on a NORMAL day: profile(t+h) x capacity."""
    h = HORIZON_MINUTES[horizon_key]
    return (X[horizon_col(h, "profile")] * X["operational_capacity"]).to_numpy(dtype=float)


class MultiHorizonForecaster:
    """
    Direct multi-horizon estimators (H=15, H=30, H=60 minutes): one model per horizon, each predicting
    the vehicle count at that horizon straight from the current state, calendar and profile features.

    (A "predict the deviation from the normal-day profile" variant was measured in
    experiments/forecast_ablation.py and was slightly worse than predicting the count directly, so it
    is not offered: the profile reaches the model as features instead.)
    """
    def __init__(self, model_type: str = "xgboost", use_pipeline: bool = True,
                 params: Dict[str, Any] = None):
        self.model_type = model_type
        self.use_pipeline = use_pipeline
        self.params = params or {}          # optional overrides on top of model_config.yaml
        self.models: Dict[str, Any] = {}
        self.horizons = ["H15", "H30", "H60"]

    def _get_estimator(self):
        """Hyperparameters come from configs/model_config.yaml."""
        if self.model_type == "ridge":
            return Ridge(**{**xgb_params("ridge"), **self.params})
        elif self.model_type == "random_forest":
            return RandomForestRegressor(**{**xgb_params("random_forest"), **self.params})
        elif self.model_type == "xgboost":
            return xgb.XGBRegressor(**{**xgb_params("xgboost_regressor"), **self.params})
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def _build(self):
        est = self._get_estimator()
        return build_pipeline(est) if self.use_pipeline else est

    def fit(self, X_train: pd.DataFrame, targets_train: Dict[str, pd.Series],
            sample_weight: Optional[np.ndarray] = None):
        """Fits dedicated models for H15, H30, and H60 (optionally weighting rows, e.g. real over synthetic)."""
        for h in self.horizons:
            estimator = self._build()
            if sample_weight is None:
                estimator.fit(X_train, np.asarray(targets_train[h], dtype=float))
            else:
                key = "model__sample_weight" if self.use_pipeline else "sample_weight"
                estimator.fit(X_train, np.asarray(targets_train[h], dtype=float), **{key: sample_weight})
            self.models[h] = estimator

    def predict_horizon(self, X: pd.DataFrame, horizon_key: str) -> np.ndarray:
        return self.models[horizon_key].predict(X)

    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generates predictions for all horizons."""
        return {h: self.predict_horizon(X, h) for h in self.horizons}

    @staticmethod
    def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        return {
            "MAE": round(mae, 3),
            "RMSE": round(rmse, 3),
            "R2": round(r2, 4)
        }
