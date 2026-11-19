import pandas as pd
from fastapi.testclient import TestClient

from api.main import app

c = TestClient(app)


def test_demand_forecast_shape_and_ranges():
    r = c.get("/api/demand-forecast", params={"days": 3, "start": "2026-11-16", "zone_id": "ZONE_MAINPARK_NORTH"})
    assert r.status_code == 200
    f = r.json()["forecast"]
    assert len(f) == 3 and all(len(d["hourly"]) == 24 for d in f)
    assert all(0 <= d["peak_occupancy"] <= 1.2 for d in f)
    mon, sat = f[0], c.get("/api/demand-forecast", params={"days": 1, "start": "2026-11-21",
                                                           "zone_id": "ZONE_MAINPARK_NORTH"}).json()["forecast"][0]
    assert mon["peak_occupancy"] > sat["peak_occupancy"] + 0.3   # academic lot: weekday >> weekend


def test_exam_scenario_raises_academic_demand():
    base = c.get("/api/demand-forecast", params={"days": 1, "start": "2026-11-17", "zone_id": "ZONE_MAINPARK_NORTH"}).json()
    exam = c.get("/api/demand-forecast", params={"days": 1, "start": "2026-11-17", "zone_id": "ZONE_MAINPARK_NORTH",
                                                 "exam_dates": "2026-11-17"}).json()
    assert exam["forecast"][0]["hours_above_90"] >= base["forecast"][0]["hours_above_90"]
    assert exam["scenario"]["exam_dates"] == ["2026-11-17"]


def test_demand_forecast_validation():
    assert c.get("/api/demand-forecast", params={"zone_id": "NOPE"}).status_code == 404
    assert c.get("/api/demand-forecast", params={"exam_dates": "not-a-date"}).status_code == 422
    assert c.get("/api/demand-forecast", params={"days": 99}).status_code == 422


def test_zone_analysis_archetypes():
    acad = c.get("/api/zone-analysis/ZONE_MAINPARK_NORTH").json()
    host = c.get("/api/zone-analysis/ZONE_GH5").json()
    assert 8 <= acad["weekday_peak_hour"] <= 16 and acad["weekend_to_weekday_ratio"] < 0.4
    assert host["weekend_to_weekday_ratio"] > 0.9                # hostel lots stay busy at weekends
    assert len(acad["weekday_profile"]) == 24 and acad["data_source"] == "simulated"
    assert c.get("/api/zone-analysis/NOPE").status_code == 404


def test_heatmaps():
    h = c.get("/api/heatmap", params={"kind": "zone_hour"}).json()
    assert len(h["rows"]) == 12 and len(h["values"][0]) == 24
    w = c.get("/api/heatmap", params={"kind": "zone_week", "zone_id": "ZONE_BLOCK1"}).json()
    assert w["rows"][0] == "Monday" and len(w["values"]) == 7
    f = c.get("/api/heatmap", params={"kind": "forecast", "date": "2026-11-17"}).json()
    assert f["data_source"] == "model_forecast" and len(f["rows"]) == 12
    assert c.get("/api/heatmap", params={"kind": "zone_week"}).status_code == 422
    assert c.get("/api/heatmap", params={"kind": "bogus"}).status_code == 422
