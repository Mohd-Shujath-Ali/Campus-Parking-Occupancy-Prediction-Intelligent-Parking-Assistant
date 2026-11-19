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


def test_history_accepts_mixed_naive_and_aware_timestamps():
    """Ingest records naive buckets while predictions use aware IST 'now'; they must interoperate."""
    from zoneinfo import ZoneInfo
    ist, utc = ZoneInfo("Asia/Kolkata"), ZoneInfo("UTC")
    h = ZoneHistory()
    h.record("ZONE_BLOCK1", datetime(2026, 11, 16, 10, 0), 40)                 # naive == IST wall clock
    assert h.lag("ZONE_BLOCK1", datetime(2026, 11, 16, 10, 15, tzinfo=ist), 15) == 40
    assert h.lag("ZONE_BLOCK1", datetime(2026, 11, 16, 4, 45, tzinfo=utc), 15) == 40   # 04:45 UTC == 10:15 IST


def test_predict_zone_after_naive_ingest_does_not_crash():
    h = ZoneHistory()
    h.record("ZONE_BLOCK1", datetime(2026, 11, 16, 9, 45), 50)                  # naive (as /api/ingest stores)
    r = predict_zone("ZONE_BLOCK1", 60, 140, timestamp=datetime(2026, 11, 16, 10, 0), history=h)
    assert r["lag_source"] in ("partial", "estimated", "observed")


def test_frozen_stream_detection():
    h = ZoneHistory()
    for k in range(6):                                  # identical readings spanning 25 minutes
        h.record("ZONE_BLOCK1", T0 + timedelta(minutes=5 * k), 77)
    assert h.is_frozen("ZONE_BLOCK1")
    h.record("ZONE_BLOCK1", T0 + timedelta(minutes=30), 79)   # feed moves again
    assert not h.is_frozen("ZONE_BLOCK1")
    quick = ZoneHistory()
    for k in range(6):                                  # identical but only seconds apart: not a freeze
        quick.record("Z", T0 + timedelta(seconds=45 * k), 12)
    assert not quick.is_frozen("Z")
