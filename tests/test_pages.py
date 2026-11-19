import pytest
from fastapi.testclient import TestClient

from api.main import app

c = TestClient(app)


@pytest.mark.parametrize("path", ["/", "/admin", "/vision", "/analytics", "/zones"])
def test_html_pages_render(path):
    """Regression: the pre-0.29 TemplateResponse(name, {'request': ...}) call 500s on current Starlette."""
    r = c.get(path)
    assert r.status_code == 200 and "<html" in r.text.lower()


def test_new_dashboard_sections_present():
    assert "heatmapTable" in c.get("/analytics").text and "outlookBody" in c.get("/analytics").text
    assert "alertList" in c.get("/admin").text


def test_snapshot_exposes_alerts_and_ranges():
    d = c.get("/api/snapshot").json()
    assert isinstance(d["alerts"], list)
    for a in d["alerts"]:
        assert a["severity"] in ("critical", "warning") and a["type"] and a["message"]
    z = d["zones"][0]
    assert z["predicted_low"] <= z["predicted_vehicle_count"] <= z["predicted_high"]


def test_metrics_include_saturation_block():
    d = c.get("/api/metrics").json()
    assert d["saturation"]["PR_AUC"] > 0.5 and d["saturation"]["calibrated"] is True
