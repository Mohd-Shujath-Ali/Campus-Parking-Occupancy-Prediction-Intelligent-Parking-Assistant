from datetime import datetime, timedelta

import pandas as pd
from fastapi.testclient import TestClient

import api.routes as routes
from api.main import app
from data_collection.store import ObservationStore, export_training_frame, floor_bucket
from models.history import ZoneHistory
from tests.auth_helpers import login_client

T = datetime(2026, 11, 16, 10, 2, 41)  # a Monday, 10:02


def test_bucket_floor():
    assert floor_bucket(T) == datetime(2026, 11, 16, 10, 0)


def test_zone_total_sums_cameras_and_latest_wins():
    st = ObservationStore(":memory:")
    st.record("ZONE_BLOCK1", "CAM_A", 30, T)
    r = st.record("ZONE_BLOCK1", "CAM_B", 12, T)
    assert r["zone_total"] == 42 and r["cameras_reporting"] == 2
    r = st.record("ZONE_BLOCK1", "CAM_A", 35, T)            # same bucket, newer reading replaces
    assert r["zone_total"] == 47 and r["cameras_reporting"] == 2


def test_export_matches_training_schema(tmp_path):
    st = ObservationStore(":memory:")
    for k in range(4):
        st.record("ZONE_GH5", "CAM_GH5_01", 50 + k, datetime(2026, 11, 16, 8, 45) + timedelta(minutes=5 * k))
    st.record("UNKNOWN_ZONE", "X", 1, T)
    out = export_training_frame(st, str(tmp_path / "live.csv"))
    ref = pd.read_csv("data/processed/lpu_timeseries.csv", nrows=1)
    assert list(out.columns) == list(ref.columns)
    assert set(out.zone_id) == {"ZONE_GH5"}                 # unknown zones dropped
    assert out.operational_capacity.iloc[0] == 160
    assert out.is_class_transition.tolist() == [1, 1, 1, 1]  # 8:45-9:00 is inside the 8:45 bell window


def _client(tmp_store, monkeypatch, key=None, as_admin=False):
    monkeypatch.setattr(routes, "_store", tmp_store)
    monkeypatch.setattr(routes, "HISTORY", ZoneHistory())
    if key:
        monkeypatch.setenv("INGEST_API_KEY", key)
    else:
        monkeypatch.delenv("INGEST_API_KEY", raising=False)
    return login_client() if as_admin else TestClient(app)


def test_ingest_endpoint_feeds_history(monkeypatch):
    c = _client(ObservationStore(":memory:"), monkeypatch, as_admin=True)
    r = c.post("/api/ingest", json={"zone_id": "ZONE_BLOCK1", "count": 55,
                                    "timestamp": "2026-11-16T10:00:00"})
    assert r.status_code == 200 and r.json()["zone_total"] == 55
    assert routes.HISTORY.lag("ZONE_BLOCK1", datetime(2026, 11, 16, 10, 15), 15) == 55
    assert c.get("/api/observations/stats").json()["rows"] == 1


def test_ingest_validation_and_auth(monkeypatch):
    c = _client(ObservationStore(":memory:"), monkeypatch, key="secret")
    body = {"zone_id": "ZONE_BLOCK1", "count": 5}
    assert c.post("/api/ingest", json=body).status_code == 401
    assert c.post("/api/ingest", json=body, headers={"X-API-Key": "secret"}).status_code == 200
    bad = {"zone_id": "NOPE", "count": 5}
    assert c.post("/api/ingest", json=bad, headers={"X-API-Key": "secret"}).status_code == 404
    assert c.post("/api/ingest", json={**body, "count": -1}, headers={"X-API-Key": "secret"}).status_code == 422
