import joblib
import numpy as np
import pytest

from models.explain import contributions
from models.inference import _build_feature_row
from models.pipeline import transformed_feature_names


def _row(zone="ZONE_BLOCK1", count=80):
    X, _ = _build_feature_row(zone, count, 140, False, False)
    return X


def test_native_contributions_are_additive():
    pipe = joblib.load("artifacts/pipelines/xgb_multi_horizon.joblib").models["H15"]
    X = _row()
    sv, base = contributions(pipe, X)
    assert len(sv) == len(transformed_feature_names(pipe))
    assert abs(base + sv.sum() - float(pipe.predict(X)[0])) < 0.05     # exact decomposition


def test_native_contributions_match_shap_library():
    shap = pytest.importorskip("shap")          # shap is research-only; not installed in the slim deploy
    pipe = joblib.load("artifacts/pipelines/xgb_multi_horizon.joblib").models["H15"]
    for zone, count in (("ZONE_BLOCK1", 80), ("ZONE_GH5", 120), ("ZONE_MAINPARK_NORTH", 30)):
        X = _row(zone, count)
        sv, base = contributions(pipe, X)
        Z = pipe.named_steps["prep"].transform(X)
        ref = shap.TreeExplainer(pipe.named_steps["model"]).shap_values(Z)[0]
        assert np.allclose(sv, ref, atol=1e-3)
