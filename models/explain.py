"""SHAP explanations for the persisted forecaster pipelines (used by the API and scripts)."""
from functools import lru_cache
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import shap

from models.pipeline import transformed_feature_names

MODEL_PATH = "artifacts/pipelines/xgb_multi_horizon.joblib"


@lru_cache(maxsize=None)
def _explainer(horizon: str):
    pipe = joblib.load(MODEL_PATH).models[horizon]
    return pipe, shap.TreeExplainer(pipe.named_steps["model"])


def _group(name: str) -> str:
    return "zone identity" if name.startswith("zone=") else name


def explain_row(X_row: pd.DataFrame, horizon: str = "H15", top_k: int = 6) -> Dict[str, Any]:
    """
    Additive explanation of one prediction: prediction = base_value + sum(contributions).
    Contributions are in cars; the 12 zone one-hot columns are merged into one.
    """
    pipe, expl = _explainer(horizon)
    Z = pipe.named_steps["prep"].transform(X_row)
    names = transformed_feature_names(pipe)
    sv = expl.shap_values(Z)[0]
    contrib: Dict[str, float] = {}
    for n, v in zip(names, sv):
        contrib[_group(n)] = contrib.get(_group(n), 0.0) + float(v)
    top = sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k]
    base = float(np.ravel(expl.expected_value)[0])
    return {
        "base_value": round(base, 2),
        "prediction": round(float(pipe.predict(X_row)[0]), 2),
        "top_contributions": [{"feature": k, "value": (None if k == "zone identity" else
                               float(X_row.iloc[0][k])), "effect_cars": round(v, 2)} for k, v in top],
        "other_effect_cars": round(sum(contrib.values()) - sum(v for _, v in top), 2),
    }
