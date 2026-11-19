import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils.config import xgb_params
from models.pipeline import build_pipeline

class MultiHorizonForecaster:
    """
    Manages direct multi-horizon estimators (H=15, H=30, H=60 minutes)
    and benchmarks them against Persistence and Historical Mean baselines.
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

    def fit(self, X_train: pd.DataFrame, targets_train: Dict[str, pd.Series]):
        """
        Fits dedicated models for H15, H30, and H60.
        """
        for h in self.horizons:
            estimator = self._build()
            estimator.fit(X_train, targets_train[h])
            self.models[h] = estimator

    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Generates predictions for all horizons.
        """
        preds = {}
        for h in self.horizons:
            preds[h] = self.models[h].predict(X)
        return preds

    @staticmethod
    def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculates MAE, RMSE, and R2.
        """
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        return {
            "MAE": round(mae, 3),
            "RMSE": round(rmse, 3),
            "R2": round(r2, 4)
        }
