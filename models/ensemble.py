"""Leak-free stacking for time series."""
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import Ridge


class BlendStackingRegressor:
    """
    Stacked generalisation with a *chronological* hold-out blend.

    sklearn's StackingRegressor builds out-of-fold predictions with shuffled or
    overlapping folds, which leaks the future in time series (and rejects
    TimeSeriesSplit because it is not a partition). Instead:

      1. fit every base pipeline on the TRAIN period,
      2. fit a non-negative Ridge meta-learner on the base predictions for the
         later VALIDATION period,
      3. evaluate on the still-later TEST period.

    The meta-learner therefore only ever learns from data that is strictly
    after the base models' training data and before the test data.
    """

    def __init__(self, base_models: Dict[str, object], alpha: float = 1.0):
        self.base_models = base_models
        self.alpha = alpha

    def _base_preds(self, X: pd.DataFrame) -> np.ndarray:
        return np.column_stack([m.predict(X) for m in self.fitted_.values()])

    def fit(self, X_train, y_train, X_val, y_val):
        self.fitted_ = {n: clone(m).fit(X_train, y_train) for n, m in self.base_models.items()}
        self.meta_ = Ridge(alpha=self.alpha, positive=True).fit(self._base_preds(X_val), y_val)
        return self

    def predict(self, X) -> np.ndarray:
        return self.meta_.predict(self._base_preds(X))

    @property
    def weights_(self) -> Dict[str, float]:
        return {k: float(round(v, 3)) for k, v in zip(self.fitted_.keys(), self.meta_.coef_)}
