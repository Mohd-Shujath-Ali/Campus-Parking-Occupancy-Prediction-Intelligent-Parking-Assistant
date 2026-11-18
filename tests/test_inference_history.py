from datetime import datetime, timedelta

from features.feature_pipeline import FEATURE_COLS
from models.history import ZoneHistory
from models.inference import _build_feature_row, predict_zone

T0 = datetime(2026, 11, 16, 10, 0)  # a Monday


def _fill(h, zone="ZONE_BLOCK1", n=13, start=20):
    for k in range(n):  # t-60 ... t  every 5 min, +2 cars per step
        h.record(zone, T0 - timedelta(minutes=5 * (n - 1 - k)), start + 2 * k)


def test_features_use_real_history():
    h = ZoneHistory(); _fill(h)
    X, src = _build_feature_row("ZONE_BLOCK1", 44, 140, False, False, T0, history=h)
    assert src == "observed"
    assert list(X.columns) == FEATURE_COLS
    assert X.loc[0, "lag_15m"] == 38 and X.loc[0, "lag_60m"] == 20
    assert X.loc[0, "delta_15m"] == 6          # real inflow, not 0
    assert X.loc[0, "rolling_std_30m"] > 0


def test_missing_history_is_flagged_estimated():
    X, src = _build_feature_row("ZONE_BLOCK1", 44, 140, False, False, T0, history=ZoneHistory())
    assert src == "estimated"
    assert X.loc[0, "delta_15m"] == 0


def test_predict_zone_reports_lag_source_and_records():
    h = ZoneHistory()
    r1 = predict_zone("ZONE_BLOCK1", 50, 140, timestamp=T0, history=h)
    assert r1["lag_source"] == "estimated"
    r2 = predict_zone("ZONE_BLOCK1", 60, 140, timestamp=T0 + timedelta(minutes=15), history=h)
    assert r2["lag_source"] in ("partial", "estimated")  # only one past point
    assert h.lag("ZONE_BLOCK1", T0 + timedelta(minutes=15), 15) == 50
