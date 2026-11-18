import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from utils.config import xgb_params
from models.pipeline import build_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

class SaturationClassifier:
    """
    Dual-Head Secondary Guard: Predicts P(Occupancy >= 90% Capacity at t+H).
    Handles class imbalance using scale_pos_weight.
    """
    def __init__(self, **overrides):
        # Defaults come from configs/model_config.yaml (xgboost_classifier)
        self.params = {**xgb_params("xgboost_classifier"), **overrides}
        self.model = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: pd.DataFrame = None, y_val: pd.Series = None):
        # Class imbalance handled with scale_pos_weight (neg/pos ratio). This
        # improves recall but inflates raw probabilities, so when a validation
        # period is supplied the scores are recalibrated (isotonic) on it.
        pos_count = (y_train == 1).sum()
        neg_count = (y_train == 0).sum()
        scale_pos_weight = float(neg_count / max(1, pos_count))

        # Same preprocessing Pipeline as the regressors, persisted as one object
        self.base = build_pipeline(
            xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, **self.params))
        self.base.fit(X_train, y_train)
        self.model = self.base
        if X_val is not None:
            # Validation is chronologically after training and before test: leak-free
            self.model = CalibratedClassifierCV(
                FrozenEstimator(self.base), method="isotonic").fit(X_val, y_val)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    @staticmethod
    def evaluate(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
        return {
            "Accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "Precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            "Recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            "F1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
            "PR_AUC": round(float(average_precision_score(y_true, y_proba)), 4) if len(np.unique(y_true)) > 1 else 0.0
        }
