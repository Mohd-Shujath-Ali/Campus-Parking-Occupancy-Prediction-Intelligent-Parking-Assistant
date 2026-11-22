import pytest
import pandas as pd
import numpy as np
from features.feature_pipeline import TimeSeriesFeaturePipeline
from data_pipeline.splitter import ChronologicalSplitter

def test_feature_pipeline_no_future_leakage():
    # Construct 10 timestamps at 5-minute intervals
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-11-15 08:00:00", periods=10, freq="5min"),
        "zone_id": "ZONE_BLOCK1",
        "vehicle_count": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "operational_capacity": 100,
        "is_exam_day": [0]*10,
        "is_class_transition": [0]*10
    })

    pipe = TimeSeriesFeaturePipeline(lag_minutes=[15], rolling_windows_minutes=[30])
    df_feat = pipe.transform_zone_series(df, cadence_minutes=5)

    # 15 min lag at step 3 (08:15) should be step 0 (08:00), which was 10
    assert df_feat.loc[3, "lag_15m"] == 10
    # Lag at step 0, 1, 2 must be NaN
    assert np.isnan(df_feat.loc[0, "lag_15m"])

    # Shifted rolling mean at step 1 should only look at step 0 (10)
    assert df_feat.loc[1, "rolling_mean_30m"] == 10

def test_chronological_splitter_gap():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-11-15 00:00:00", periods=100, freq="1h"),
        "zone_id": "ZONE_TEST",
        "vehicle_count": list(range(100))
    })

    splitter = ChronologicalSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, buffer_hours=2)
    df_train, df_val, df_test = splitter.split_dataframe(df)

    # Check temporal order: max(train) + buffer <= min(val)
    train_max = df_train['timestamp'].max()
    val_min = df_val['timestamp'].min()
    assert (val_min - train_max).total_seconds() >= 2 * 3600

    val_max = df_val['timestamp'].max()
    test_min = df_test['timestamp'].min()
    assert (test_min - val_max).total_seconds() >= 2 * 3600


def _two_zone_frame():
    ts = pd.date_range("2026-11-15 08:00:00", periods=20, freq="5min")
    a = pd.DataFrame({"timestamp": ts, "zone_id": "ZONE_BLOCK1", "vehicle_count": range(20),
                      "operational_capacity": 100, "is_exam_day": 0, "is_class_transition": 0})
    b = pd.DataFrame({"timestamp": ts, "zone_id": "ZONE_GH5", "vehicle_count": range(1000, 1020),
                      "operational_capacity": 160, "is_exam_day": 0, "is_class_transition": 0})
    return pd.concat([a, b]).sort_values("timestamp").reset_index(drop=True)


def test_zone_identity_kept_and_no_cross_zone_bleed():
    from features.feature_pipeline import zone_code
    out = TimeSeriesFeaturePipeline().transform_zone_series(_two_zone_frame())
    assert set(out["zone_id"]) == {"ZONE_BLOCK1", "ZONE_GH5"}
    assert (out.loc[out.zone_id == "ZONE_BLOCK1", "zone_code"] == zone_code("ZONE_BLOCK1")).all()
    assert zone_code("ZONE_BLOCK1") != zone_code("ZONE_GH5")
    # lags of zone BLOCK1 must come from BLOCK1 only (counts < 100)
    assert out.loc[out.zone_id == "ZONE_BLOCK1", "lag_60m"].dropna().max() < 100


def test_training_matrix_matches_shared_schema():
    from features.feature_pipeline import FEATURE_COLS
    pipe = TimeSeriesFeaturePipeline()
    X, targets = pipe.prepare_training_matrices(
        pipe.transform_zone_series(_two_zone_frame().assign(is_campus_event=0)))
    assert list(X.columns) == FEATURE_COLS
    assert len(X) == len(targets["H15"])
