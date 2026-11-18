import numpy as np
import pandas as pd
from features.feature_pipeline import TimeSeriesFeaturePipeline
from models.baselines import SeasonalHistoricalMeanForecaster, PersistenceForecaster


def _frame():
    ts = pd.date_range("2026-09-01", periods=7 * 288, freq="5min")  # 1 week
    parts = []
    for z, level in (("ZONE_BLOCK1", 10), ("ZONE_GH5", 100)):
        parts.append(pd.DataFrame({"timestamp": ts, "zone_id": z, "vehicle_count": level,
                                   "operational_capacity": 200, "is_exam_day": 0,
                                   "is_class_transition": 0}))
    return pd.concat(parts)


def test_seasonal_baseline_is_zone_aware():
    pipe = TimeSeriesFeaturePipeline()
    X, t = pipe.prepare_training_matrices(pipe.transform_zone_series(_frame()))
    pred = SeasonalHistoricalMeanForecaster().fit(X, t["H15"]).predict(X)
    # Two zones with constant 10 / 100 cars must not be averaged together
    assert np.allclose(pred[X["vehicle_count"].values == 10], 10)
    assert np.allclose(pred[X["vehicle_count"].values == 100], 100)


def test_persistence_returns_current_count():
    X = pd.DataFrame({"vehicle_count": [3, 7]})
    assert list(PersistenceForecaster().predict(X)) == [3, 7]
