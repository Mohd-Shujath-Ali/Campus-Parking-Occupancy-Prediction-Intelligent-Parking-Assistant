"""
API Integration Tests — LPU Campus Parking Intelligence System
Tests verify real API contracts, ML model integration, and data integrity.
"""
import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


# ─────────────────────────── Health ───────────────────────────

def test_health_endpoint():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["total_monitored_zones"] == 12
    assert "ml_model" in data
    assert "yolo_model" in data
    assert "timestamp" in data


# ─────────────────────────── Zones ────────────────────────────

def test_zones_endpoint_count_and_structure():
    res = client.get("/api/zones")
    assert res.status_code == 200
    zones = res.json()
    assert len(zones) == 12
    zone_ids = {z["zone_id"] for z in zones}
    assert "ZONE_BLOCK1" in zone_ids
    assert "ZONE_MAINPARK_NORTH" in zone_ids
    # Verify required fields are present
    for z in zones:
        for field in ("zone_id", "name", "category", "operational_capacity", "camera_id"):
            assert field in z, f"Missing field '{field}' in zone {z.get('zone_id')}"


# ─────────────────────────── Predict (ML) ─────────────────────

def test_prediction_endpoint_basic():
    payload = {
        "zone_id": "ZONE_BLOCK1",
        "horizon_minutes": 15,
        "current_count": 110,
        "is_exam_day": False,
        "is_class_transition": True
    }
    res = client.post("/api/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["zone_id"] == "ZONE_BLOCK1"
    assert isinstance(data["predicted_vehicle_count"], int)
    assert data["predicted_vehicle_count"] >= 0
    assert 0.0 <= data["occupancy_rate"] <= 1.5   # allow mild overflow
    assert data["saturation_probability"] >= 0.0
    assert data["status"] in ("AVAILABLE", "FILLING_FAST", "SATURATED")
    assert "available_spaces" in data
    assert data["confidence_level"] in ("HIGH_CONFIDENCE_ML", "HEURISTIC_FALLBACK")


def test_prediction_all_horizons():
    base = {"zone_id": "ZONE_MAINPARK_NORTH", "current_count": 200,
            "is_exam_day": False, "is_class_transition": False}
    for h in (15, 30, 60):
        res = client.post("/api/predict", json={**base, "horizon_minutes": h})
        assert res.status_code == 200, f"Horizon {h} failed: {res.text}"
        assert res.json()["horizon_minutes"] == h


def test_prediction_invalid_zone():
    res = client.post("/api/predict", json={
        "zone_id": "ZONE_NONEXISTENT",
        "horizon_minutes": 15,
        "current_count": 50
    })
    assert res.status_code == 404


def test_prediction_invalid_horizon():
    res = client.post("/api/predict", json={
        "zone_id": "ZONE_BLOCK1",
        "horizon_minutes": 45,   # only 15/30/60 allowed
        "current_count": 50
    })
    assert res.status_code == 422


def test_prediction_negative_count():
    res = client.post("/api/predict", json={
        "zone_id": "ZONE_BLOCK1",
        "horizon_minutes": 15,
        "current_count": -1
    })
    assert res.status_code == 422


# ─────────────────────────── Recommendations ──────────────────

def test_recommendations_endpoint_structure():
    res = client.get("/api/recommendations?destination=Block_26_CSE&horizon_minutes=15")
    assert res.status_code == 200
    data = res.json()
    assert data["destination"] == "Block_26_CSE"
    assert isinstance(data["recommended_zones"], list)
    assert len(data["recommended_zones"]) == 12
    # Verify each zone has required fields
    for z in data["recommended_zones"]:
        for field in ("zone_id", "name", "score", "available_spaces", "occupancy_rate",
                      "saturation_probability", "walking_distance_meters", "status"):
            assert field in z, f"Missing '{field}' in zone recommendation"


def test_recommendations_scores_are_numeric():
    res = client.get("/api/recommendations?destination=Block_1&horizon_minutes=30")
    assert res.status_code == 200
    zones = res.json()["recommended_zones"]
    for z in zones:
        assert isinstance(z["score"], (int, float))
        assert 0.0 <= z["occupancy_rate"] <= 1.5


def test_recommendations_invalid_destination():
    res = client.get("/api/recommendations?destination=NONEXISTENT_DEST")
    assert res.status_code == 404


# ─────────────────────────── Snapshot ─────────────────────────

def test_snapshot_endpoint():
    res = client.get("/api/snapshot")
    assert res.status_code == 200
    data = res.json()
    assert "total_capacity" in data
    assert "total_occupied" in data
    assert "campus_occupancy_rate" in data
    assert isinstance(data["zones"], list)
    assert len(data["zones"]) == 12
    assert data["total_capacity"] > 0
    # Campus occupancy rate must be valid proportion
    assert 0.0 <= data["campus_occupancy_rate"] <= 1.0


def test_snapshot_zone_fields():
    res = client.get("/api/snapshot")
    zones = res.json()["zones"]
    for z in zones:
        for f in ("zone_id", "name", "camera_id", "operational_capacity",
                  "predicted_vehicle_count", "available_spaces", "occupancy_rate",
                  "saturation_probability", "status", "camera_age_seconds"):
            assert f in z, f"Snapshot zone missing field '{f}'"


# ─────────────────────────── Metrics ──────────────────────────

def test_metrics_endpoint():
    res = client.get("/api/metrics")
    assert res.status_code == 200
    data = res.json()
    assert "models" in data
    assert isinstance(data["models"], list)
    assert len(data["models"]) >= 3  # at least Persistence, Seasonal Mean, XGBoost
    # Verify XGBoost entry exists
    xgb_rows = [m for m in data["models"] if m.get("Model") == "Tuned XGBoost"]
    assert len(xgb_rows) >= 1


# ─────────────────────────── YOLO Detection ───────────────────

def test_detect_image_endpoint():
    test_imgs = list(Path("collab_content_backup/LPU---Parking-1/test/images").glob("*.jpg"))
    if not test_imgs:
        pytest.skip("No test images available")
    with open(test_imgs[0], "rb") as f:
        res = client.post("/api/detect-image", files={"file": ("parking.jpg", f, "image/jpeg")})
    assert res.status_code == 200
    data = res.json()
    assert "vehicle_count" in data
    assert "annotated_image_base64" in data
    assert data["vehicle_count"] >= 0
    assert isinstance(data["annotated_image_base64"], str)
    assert len(data["annotated_image_base64"]) > 100  # non-empty base64


def test_detect_image_empty_file():
    from io import BytesIO
    res = client.post("/api/detect-image", files={"file": ("empty.jpg", BytesIO(b""), "image/jpeg")})
    assert res.status_code == 400


def test_detect_frame_empty():
    from io import BytesIO
    res = client.post("/api/detect-frame", files={"file": ("frame.jpg", BytesIO(b""), "image/jpeg")})
    assert res.status_code == 400


# ─────────────────────────── Favicon ──────────────────────────

def test_favicon_served():
    res = client.get("/favicon.ico")
    assert res.status_code == 200
    assert res.headers["content-type"] in ("image/x-icon", "image/vnd.microsoft.icon",
                                           "image/ico", "application/octet-stream")


def test_explain_endpoint_is_additive():
    r = client.get("/api/explain", params={"zone_id": "ZONE_BLOCK1", "current_count": 80,
                                           "horizon_minutes": 15})
    assert r.status_code == 200
    d = r.json()
    total = d["base_value"] + sum(c["effect_cars"] for c in d["top_contributions"]) + d["other_effect_cars"]
    assert abs(total - d["prediction"]) < 0.5      # SHAP is additive (rounding only)
    assert d["top_contributions"][0]["feature"]


def test_explain_unknown_zone_404():
    assert client.get("/api/explain", params={"zone_id": "NOPE", "current_count": 1}).status_code == 404


def test_predict_returns_interval_containing_point_forecast():
    r = client.post("/api/predict", json={"zone_id": "ZONE_BLOCK1", "current_count": 90,
                                          "horizon_minutes": 30})
    d = r.json()
    assert d["predicted_low"] is not None
    assert d["predicted_low"] <= d["predicted_vehicle_count"] <= d["predicted_high"]


# ─────────────────────────── Anti-Herding & Navigation ────────

def test_navigate_endpoint():
    res = client.post("/api/navigate", json={"zone_id": "ZONE_BLOCK1"})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "DISPATCH_REGISTERED"
    assert data["zone_id"] == "ZONE_BLOCK1"
    assert data["active_in_flight_count"] >= 1


def test_navigate_invalid_zone():
    res = client.post("/api/navigate", json={"zone_id": "ZONE_UNKNOWN_XYZ"})
    assert res.status_code == 404


# ─────────────────────────── SHAP & Figures ───────────────────

def test_shap_figure_endpoint():
    res = client.get("/api/shap-figure/shap_summary_beeswarm.png")
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/png"
    assert len(res.content) > 1000


def test_shap_figure_whitelist_guard():
    res = client.get("/api/shap-figure/unauthorized_file.png")
    assert res.status_code == 404


# ─────────────────────────── CV Quality & Edge Guard ──────────

def test_image_quality_assessment():
    import numpy as np
    from computer_vision.detection import assess_image_quality

    # Underexposed (black frame)
    black_img = np.zeros((100, 100, 3), dtype=np.uint8)
    q_black = assess_image_quality(black_img)
    assert q_black["is_degraded"] is True
    assert any("underexposure" in r.lower() for r in q_black["reasons"])

    # Normal illuminated frame
    normal_img = np.random.randint(60, 180, (100, 100, 3), dtype=np.uint8)
    q_norm = assess_image_quality(normal_img)
    assert q_norm["mean_brightness"] >= 50.0


# ─────────────────────────── Edge Case Hardening ──────────────

def test_prediction_naive_iso_timestamp():
    """Verify offset-naive timestamps do not trigger datetime subtraction exceptions."""
    payload = {
        "zone_id": "ZONE_BLOCK1",
        "horizon_minutes": 15,
        "current_count": 85,
        "timestamp": "2026-11-18T10:00:00"  # Note: No +05:30 offset
    }
    res = client.post("/api/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["predicted_vehicle_count"] >= 0


def test_recommendation_fuzzy_destination():
    """Verify space/case insensitive destination resolution."""
    for dest_query in ("block 26 cse", "BLOCK_26_CSE", "Block 26 CSE"):
        res = client.get(f"/api/recommendations?destination={dest_query}&horizon_minutes=15")
        assert res.status_code == 200, f"Failed for {dest_query}: {res.text}"
        data = res.json()
        assert data["destination"] == "Block_26_CSE"
        assert len(data["recommended_zones"]) > 0


def test_navigate_capacity_cap():
    """Verify repeated navigation dispatches don't infinitely starve capacity beyond 35%."""
    # Zone ZONE_BLOCK1 has operational capacity 150 -> 35% cap is 52
    for _ in range(60):
        res = client.post("/api/navigate", json={"zone_id": "ZONE_BLOCK1"})
        assert res.status_code in (200, 429)  # 200 within limit, or 429 once rate limit reached
        if res.status_code == 200:
            assert res.json()["active_in_flight_count"] <= 55
