"""
Per-prediction explanations for the persisted forecaster pipelines (used by /api/explain).

Contributions are exact TreeSHAP values computed by XGBoost itself (`pred_contribs=True`), so the
serving path needs neither the `shap` package nor numba (which add ~190 MB of RAM). The values are
identical to shap.TreeExplainer's for tree models; tests/test_explain.py checks this when `shap`
is installed.
"""
from typing import Any, Dict

import numpy as np
import pandas as pd
import xgboost as xgb

from models import inference
from models.pipeline import transformed_feature_names


def _pipeline(horizon: str):
    """The forecaster already loaded for serving (a second joblib.load would duplicate it in RAM)."""
    if not inference._load_models():
        raise RuntimeError(inference._load_error or "Model artifacts not available")
    return inference._xgb_model.models[horizon]


def _group(name: str) -> str:
    return "zone identity" if name.startswith("zone=") else name


def contributions(pipe, X_row: pd.DataFrame):
    """(per-feature contributions in transformed space, base value) via XGBoost's native TreeSHAP."""
    Z = pipe.named_steps["prep"].transform(X_row)
    booster = pipe.named_steps["model"].get_booster()
    contribs = booster.predict(xgb.DMatrix(np.asarray(Z, dtype=np.float32)), pred_contribs=True)
    return contribs[0, :-1], float(contribs[0, -1])


def explain_row(X_row: pd.DataFrame, horizon: str = "H15", top_k: int = 6) -> Dict[str, Any]:
    """
    Additive explanation of one prediction: prediction = base_value + sum(contributions).
    Contributions are in cars; the 12 zone one-hot columns are merged into one.
    """
    pipe = _pipeline(horizon)
    sv, base = contributions(pipe, X_row)
    contrib: Dict[str, float] = {}
    for n, v in zip(transformed_feature_names(pipe), sv):
        contrib[_group(n)] = contrib.get(_group(n), 0.0) + float(v)
    top = sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k]
    return {
        "base_value": round(base, 2),
        "prediction": round(float(pipe.predict(X_row)[0]), 2),
        "top_contributions": [{"feature": k, "value": (None if k == "zone identity" else
                               float(X_row.iloc[0][k])), "effect_cars": round(v, 2)} for k, v in top],
        "other_effect_cars": round(sum(contrib.values()) - sum(v for _, v in top), 2),
    }
