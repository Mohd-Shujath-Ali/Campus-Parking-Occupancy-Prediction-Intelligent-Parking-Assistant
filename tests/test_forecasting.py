"""
Forecasting stack: calendar, profile, features (train/serve parity), uncertainty, horizon interpolation,
the live data layer (demo / camera / stale / estimate), calendar API, model monitoring and the recommender.
"""
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import calendar_store, model_monitor
from api.db import get_db
from api.main import app
from api.routes import ZONES_MAP
from data_collection.store import ObservationStore
from data_pipeline.generator import synthetic_calendar
from features import calendar as cal
from features.calendar import CalendarEntry, compute_flags, day_range, event_ratio_by_zone, zone_ids
from features.feature_pipeline import FEATURE_COLS, HORIZONS, TimeSeriesFeaturePipeline, horizon_col
from features.profile import ProfileModel
from models import inference, live
from models.history import HISTORY, ZoneHistory
from tests.auth_helpers import client_for_role, login_client

NOW = datetime(2026, 11, 17, 10, 7)          # a Tuesday morning, naive campus time
CAPS = {z: m["operational_capacity"] for z, m in ZONES_MAP.items()}


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    """Every test starts with an empty calendar, no live counts and the demo feed allowed."""
    def wipe():
        with get_db().conn() as c:
            c.execute("DELETE FROM calendar_entries")
            c.execute("DELETE FROM prediction_log")
        calendar_store._invalidate()
        HISTORY.clear()
        live.bump()
    wipe()
    monkeypatch.setattr(live, "_ever_real", False)
    monkeypatch.delenv("PARKING_DEMO_FEED", raising=False)
    yield
    wipe()


# ─────────────────────────── Calendar ───────────────────────────

def _event(venue="Block_1", vehicles=100, start=NOW, hours=3):
    return CalendarEntry("event", "Test event", start, start + timedelta(hours=hours), venue, vehicles)


def test_event_load_is_shared_by_distance_and_conserves_vehicles():
    ids = zone_ids()
    w = cal.event_zone_weights("Block_1")                              # each ground's share of the event's vehicles
    assert w[ids.index("ZONE_BLOCK1")] > w[ids.index("ZONE_MAINGATE")] > w[ids.index("ZONE_BLOCK57")]
    assert w.sum() == pytest.approx(1.0)
    r = event_ratio_by_zone(_event("Block_1", 100))                    # as an occupancy ratio, i.e. per slot of capacity
    assert sum(r[i] * CAPS[z] for i, z in enumerate(ids)) == pytest.approx(100, rel=1e-6)


def test_flags_exam_holiday_event_and_bell():
    entries = [day_range(date(2026, 11, 16), date(2026, 11, 22), "exam", "Exams"),
               day_range(date(2026, 11, 18), date(2026, 11, 18), "holiday", "Holiday"),
               _event("Block_1", 100, datetime(2026, 11, 17, 9, 0), 4)]
    ts = [datetime(2026, 11, 17, 8, 50), datetime(2026, 11, 17, 12, 0), datetime(2026, 11, 18, 9, 0),
          datetime(2026, 11, 21, 9, 0)]                              # Tue, Tue, Wed(holiday), Sat
    f = compute_flags(ts, "ZONE_BLOCK1", entries)
    assert f.is_exam_day.tolist() == [1, 1, 0, 0]                    # not on the holiday, not on the Saturday
    assert f.is_holiday.tolist() == [0, 0, 1, 0]
    assert f.is_class_transition.tolist() == [1, 0, 0, 0]            # bell at 8:45; none on a holiday or weekend
    assert f.is_campus_event.tolist() == [0, 1, 0, 0] and f.event_load[1] > 0 and f.event_load[0] == 0
    assert compute_flags(ts[:2], "ZONE_MAINGATE", entries).event_load[1] < f.event_load[1]   # farther from the venue


def test_context_describes_now():
    entries = [_event("Block_1", 100, NOW - timedelta(hours=1), 3),
               CalendarEntry("event", "Later", "x" and NOW + timedelta(hours=2), NOW + timedelta(hours=4), "Law_Block", 50)]
    ctx = cal.context_at(NOW, entries)
    assert [e["name"] for e in ctx["events_now"]] == ["Test event"] and [e["name"] for e in ctx["events_soon"]] == ["Later"]


# ─────────────────────────── Profile ───────────────────────────

def _profile_frame():
    rows = []
    for week, level in enumerate((0.2, 0.4, 0.9)):
        for day in range(5):
            d = datetime(2026, 8, 3) + timedelta(days=7 * week + day)
            for minute in (0, 5, 10):
                rows.append({"timestamp": d.replace(hour=10, minute=minute), "zone_id": "ZONE_BLOCK1",
                             "vehicle_count": level * 100, "operational_capacity": 100, "is_exam_day": 0})
    return pd.DataFrame(rows)


def test_profile_is_leave_one_week_out_for_training_rows():
    p = ProfileModel().fit(_profile_frame())
    ts = [datetime(2026, 8, 17, 10, 5)]                              # a week-2 Monday
    zc = [zone_ids().index("ZONE_BLOCK1")]
    assert p.lookup(zc, ts)[0] == pytest.approx(0.5)                 # mean of all three weeks
    assert p.lookup(zc, ts, leave_week_out=True)[0] == pytest.approx(0.3)   # own week removed: (0.2 + 0.4) / 2


def test_profile_ignores_exam_and_event_days():
    df = _profile_frame()
    exam = df.copy()
    exam["timestamp"] += pd.Timedelta(days=21)
    exam["vehicle_count"], exam["is_exam_day"] = 100, 1               # full lots, but on exam days
    p = ProfileModel().fit(pd.concat([df, exam]))
    assert p.lookup([zone_ids().index("ZONE_BLOCK1")], [datetime(2026, 8, 17, 10, 5)])[0] == pytest.approx(0.5)


# ─────────────────────────── Features ───────────────────────────

def test_gap_in_observations_does_not_misalign_lags_or_targets():
    ts = pd.date_range("2026-11-17 08:00", periods=30, freq="5min")
    df = pd.DataFrame({"timestamp": ts, "zone_id": "ZONE_BLOCK1", "vehicle_count": np.arange(30) * 2,
                       "operational_capacity": 100, "is_exam_day": 0, "is_class_transition": 0})
    gapped = df.drop(index=range(10, 14)).reset_index(drop=True)      # camera outage: 20 minutes missing
    out = TimeSeriesFeaturePipeline().transform_zone_series(gapped)
    row = out[out.timestamp == ts[16]].iloc[0]
    assert row.lag_15m == 2 * 13 or pd.isna(row.lag_15m)             # lag lands in the outage -> missing, never a wrong row
    assert pd.isna(row.lag_15m)
    ok = out[out.timestamp == ts[20]].iloc[0]
    assert ok.lag_15m == 2 * 17 and ok.target_count_t15 == 2 * 23     # intact stretches stay exact


def test_future_covariates_match_the_calendar():
    entries = synthetic_calendar()
    ts = pd.date_range("2026-11-26 12:00", periods=60, freq="5min")  # Thursday: exam window + Guest Lecture 14:00-17:00
    flags = compute_flags(ts, "ZONE_BLOCK57", entries)
    df = pd.DataFrame({"timestamp": ts, "zone_id": "ZONE_BLOCK57", "vehicle_count": 50, "operational_capacity": 185,
                       "is_class_transition": flags.is_class_transition, "is_exam_day": flags.is_exam_day,
                       "is_campus_event": flags.is_campus_event, "is_holiday": flags.is_holiday, "event_load": flags.event_load})
    out = TimeSeriesFeaturePipeline().transform_zone_series(df)
    row = out.iloc[0]                                                 # 12:00; the event starts at 14:00
    assert row.event_load == 0 and row[horizon_col(60, "event_load")] == 0 and row[horizon_col(60, "is_exam_day")] == 1
    row2 = out[out.timestamp == pd.Timestamp("2026-11-26 13:30")].iloc[0]
    assert row2[horizon_col(30, "event_load")] > 0 and row2.event_load == 0


def test_serving_features_equal_training_features():
    """The live feature builder must reproduce the training pipeline exactly (no train/serve skew)."""
    inference._load_models()
    prof = inference._profile
    assert prof is not None
    entries = synthetic_calendar()
    df = pd.read_csv("data/processed/lpu_timeseries.csv", parse_dates=["timestamp"])
    zone = "ZONE_BLOCK57"
    window = df[(df.zone_id == zone) & (df.timestamp >= "2026-11-26 00:00") & (df.timestamp <= "2026-11-27 04:00")]
    window = window.reset_index(drop=True)
    when = pd.Timestamp("2026-11-26 15:30")                           # exam day, inside an event, near the 14:45 bell
    train_row = TimeSeriesFeaturePipeline(profile=prof).transform_zone_series(window)
    train_row = train_row[train_row.timestamp == when].iloc[0]

    hist = ZoneHistory()
    for r in window[window.timestamp <= when].itertuples():
        hist.record(zone, r.timestamp.to_pydatetime(), int(r.vehicle_count))
    cur = int(window[window.timestamp == when].vehicle_count.iloc[0])
    X, src = inference._build_feature_row(zone, cur, CAPS[zone], None, None, when.to_pydatetime(),
                                          history=hist, entries=entries)
    assert src == "observed"
    for c in FEATURE_COLS:
        assert float(X.loc[0, c]) == pytest.approx(float(train_row[c]), abs=1e-3), c


# ─────────────────────────── Uncertainty and horizons ───────────────────────────

def test_free_slot_probability_falls_as_forecast_count_rises():
    inference._load_models()
    pf = inference._resid_model.p_free("H30", np.array([40, 100, 125, 138, 150.0]), np.full(5, 140.0), np.zeros(5))
    assert (np.diff(pf) <= 1e-9).all() and pf[0] > 0.95 and pf[-1] < 0.1 and ((pf >= 0) & (pf <= 1)).all()


def test_interval_contains_forecast_and_event_rows_get_wider_bands():
    inference._load_models()
    r = inference._resid_model
    p, cap = np.full(2, 80.0), np.full(2, 140.0)
    lo, hi = r.interval("H60", p, cap, np.zeros(2), 0.8, event=np.array([0, 1]))
    assert (lo <= p).all() and (hi >= p).all()
    assert (hi[1] - lo[1]) > (hi[0] - lo[0])                          # events are less predictable


def _fc():
    return {"capacity": 100, "current_count": 50.0, "pts": {15: 60.0, 30: 70.0, 60: 80.0},
            "low": {15: 55.0, 30: 62.0, 60: 70.0}, "high": {15: 65.0, 30: 78.0, 60: 90.0},
            "p_free": {15: 0.9, 30: 0.7, 60: 0.5}, "p_sat": {15: 0.0, 30: 0.1, 60: 0.2},
            "profile": {15: 55.0, 30: 55.0, 60: 55.0}}


def test_horizon_interpolation():
    fc = _fc()
    at = lambda h: inference.at_horizon(fc, h)
    assert at(0)["predicted"] == 50 and at(15)["predicted"] == 60 and at(60)["predicted"] == 80
    assert at(7.5)["predicted"] == pytest.approx(55) and at(45)["predicted"] == pytest.approx(75)
    assert at(45)["p_free"] == pytest.approx(0.6)
    far = at(150)                                                     # beyond +60: deviation decays back to the normal day
    assert 55 < far["predicted"] < 80 and far["high"] - far["low"] > at(60)["high"] - at(60)["low"]
    assert all(at(h)["low"] <= at(h)["predicted"] <= at(h)["high"] for h in (0, 10, 30, 50, 90))


# ─────────────────────────── Live data layer ───────────────────────────

def test_demo_feed_is_used_only_before_any_real_count_and_is_labelled():
    p = live.predict_all(ZONES_MAP, NOW, 15)
    assert {v["count_source"] for v in p.values()} == {"simulated"}
    assert {v["confidence"] for v in p.values()} == {"demo"} and {v["camera_status"] for v in p.values()} == {"demo"}
    assert all(0 <= v["current_count"] <= 1.15 * CAPS[z] for z, v in p.items())
    assert live.data_mode(ZONES_MAP)["mode"] == "demo"


def test_real_counts_replace_demo_and_never_fall_back_to_it():
    HISTORY.record("ZONE_BLOCK1", NOW - timedelta(minutes=3), 90)
    live.note_real_data()
    p = live.predict_all(ZONES_MAP, NOW, 15)
    assert p["ZONE_BLOCK1"]["count_source"] == "camera" and p["ZONE_BLOCK1"]["current_count"] == 90
    assert p["ZONE_BLOCK1"]["camera_status"] == "live"
    others = {v["count_source"] for z, v in p.items() if z != "ZONE_BLOCK1"}
    assert others == {"profile"}                                      # estimates, never invented demo numbers
    assert {v["camera_status"] for z, v in p.items() if z != "ZONE_BLOCK1"} == {"offline"}
    assert live.data_mode(ZONES_MAP, NOW)["mode"] == "live"


def test_stale_and_dead_cameras():
    live.note_real_data()
    HISTORY.record("ZONE_BLOCK1", NOW - timedelta(minutes=20), 70)
    stale = live.predict_all(ZONES_MAP, NOW, 15)["ZONE_BLOCK1"]
    assert stale["count_source"] == "camera" and stale["camera_status"] == "stale" and stale["confidence"] != "high"
    HISTORY.clear()
    HISTORY.record("ZONE_BLOCK1", NOW - timedelta(minutes=45), 70)
    live.bump()                                                       # what the ingest endpoint does after every count
    dead = live.predict_all(ZONES_MAP, NOW, 15)["ZONE_BLOCK1"]
    assert dead["count_source"] == "profile" and dead["confidence"] == "low"


def test_frozen_feed_lowers_confidence():
    live.note_real_data()
    for k in range(6):                                                # identical readings over 25 minutes
        HISTORY.record("ZONE_BLOCK1", NOW - timedelta(minutes=27 - 5 * k), 77)
    p = live.predict_all(ZONES_MAP, NOW, 15)["ZONE_BLOCK1"]
    assert p["stream_frozen"] is True and p["confidence"] == "low"


def test_reading_age_extends_the_forecast_horizon():
    """A 6-minute-old reading forecasting 'in 15 minutes' really forecasts 21 minutes past that reading."""
    live.note_real_data()
    HISTORY.record("ZONE_MAINPARK_NORTH", NOW - timedelta(minutes=6), 150)
    fcs = live.forecast_all(ZONES_MAP, NOW)
    f = fcs["ZONE_MAINPARK_NORTH"]
    direct = inference.at_horizon(f.fc, 15 + f.state.age_s / 60)["predicted"]
    assert live.describe(f, 15)["predicted_vehicle_count"] == int(round(direct))
    assert f.state.age_s == pytest.approx(360)


def test_admin_event_changes_demo_counts_and_forecasts():
    before = live.predict_all(ZONES_MAP, NOW, 30)["ZONE_BLOCK1"]
    calendar_store.add_entry("event", "Convocation", NOW - timedelta(hours=1), NOW + timedelta(hours=3), "Block_1", 300)
    live.bump()
    after = live.predict_all(ZONES_MAP, NOW, 30)["ZONE_BLOCK1"]
    assert after["current_count"] >= before["current_count"] + 10
    assert after["predicted_vehicle_count"] > before["predicted_vehicle_count"]


def test_events_exams_move_the_point_forecast_in_predict_zone():
    base = inference.predict_zone("ZONE_BLOCK57", 60, 185, 60, timestamp=NOW, history=ZoneHistory(), record=False)
    calendar_store.add_entry("event", "Guest lecture", NOW, NOW + timedelta(hours=2), "Block_57_Agri", 200)
    ev = inference.predict_zone("ZONE_BLOCK57", 60, 185, 60, timestamp=NOW, history=ZoneHistory(), record=False)
    assert ev["predicted_vehicle_count"] > base["predicted_vehicle_count"] + 5
    assert ev["p_free"] < base["p_free"]


# ─────────────────────────── Calendar API ───────────────────────────

def test_calendar_api_roles_validation_and_effect():
    admin, student = login_client(), client_for_role("student", username="cal_student", home_block="Block_1")
    body = {"kind": "event", "name": "Tech Fest", "start": "2026-12-01T09:00", "end": "2026-12-01T17:00",
            "venue_block": "Block_26_CSE", "expected_vehicles": 300}
    assert student.post("/api/calendar", json=body).status_code == 403
    assert student.get("/api/calendar").status_code == 200 and student.get("/api/calendar/context").status_code == 200
    prev = admin.post("/api/calendar/preview", json=body).json()["entry"]
    assert prev["impact"][0]["extra_vehicles"] > 0 and sum(i["extra_vehicles"] for i in prev["impact"]) <= 300
    made = admin.post("/api/calendar", json=body)
    assert made.status_code == 201 and made.json()["id"]
    assert any(e["name"] == "Tech Fest" for e in student.get("/api/calendar?days_ahead=365").json()["entries"])
    assert admin.delete(f"/api/calendar/{made.json()['id']}").status_code == 200
    assert admin.delete(f"/api/calendar/{made.json()['id']}").status_code == 404
    assert admin.get("/api/audit?action=calendar_entry_created&limit=3").json()["entries"]


@pytest.mark.parametrize("patch,why", [
    ({"venue_block": None}, "venue"), ({"venue_block": "Mars"}, "venue"), ({"expected_vehicles": None}, "vehicles"),
    ({"expected_vehicles": 0}, "vehicles"), ({"end": "2026-12-01T08:00"}, "end before start"),
    ({"end": "2026-12-09T09:00"}, "too long"), ({"start": "2026-12-01"}, "needs a time"), ({"name": "x"}, "name"),
    ({"kind": "party"}, "kind")])
def test_calendar_event_validation(patch, why):
    body = {"kind": "event", "name": "Tech Fest", "start": "2026-12-01T09:00", "end": "2026-12-01T17:00",
            "venue_block": "Block_26_CSE", "expected_vehicles": 300, **patch}
    assert login_client().post("/api/calendar", json=body).status_code == 422, why


def test_exam_and_holiday_entries_and_demand_outlook_uses_them():
    admin = login_client()
    assert admin.post("/api/calendar", json={"kind": "exam", "name": "Mid-terms", "start": "2026-12-07",
                                             "end": "2026-12-12"}).status_code == 201
    assert admin.post("/api/calendar", json={"kind": "holiday", "name": "Break", "start": "2026-12-14"}).status_code == 201
    assert admin.post("/api/calendar", json={"kind": "exam", "name": "Too long", "start": "2026-12-01",
                                             "end": "2027-03-01"}).status_code == 422
    out = admin.get("/api/demand-forecast", params={"days": 10, "start": "2026-12-07", "zone_id": "ZONE_MAINPARK_NORTH"}).json()
    assert {e["name"] for e in out["calendar_entries_applied"]} == {"Mid-terms", "Break"}
    by_day = {d["date"]: d for d in out["forecast"]}
    assert by_day["2026-12-14"]["peak_occupancy"] < by_day["2026-12-08"]["peak_occupancy"] - 0.3   # holiday: weekend-like


def test_predict_endpoint_uses_the_calendar_unless_overridden():
    admin = login_client()
    req = {"zone_id": "ZONE_MAINPARK_NORTH", "horizon_minutes": 60, "current_count": 150, "timestamp": "2026-12-08T08:30:00"}
    normal = admin.post("/api/predict", json=req).json()
    admin.post("/api/calendar", json={"kind": "exam", "name": "Exams", "start": "2026-12-08", "end": "2026-12-08"})
    exam = admin.post("/api/predict", json=req).json()
    assert exam["predicted_vehicle_count"] > normal["predicted_vehicle_count"]
    forced_off = admin.post("/api/predict", json={**req, "is_exam_day": False}).json()
    assert forced_off["predicted_vehicle_count"] <= exam["predicted_vehicle_count"]


# ─────────────────────────── Recommendations ───────────────────────────

def test_recommendation_response_carries_free_slot_chance_range_and_data_mode():
    c = client_for_role("student", username="rec_student", home_block="Block_26_CSE")
    r = c.get("/api/recommendations?horizon_minutes=30").json()
    assert r["data_mode"] == "demo" and "bell_window" in r["conditions"]
    for z in r["recommended_zones"]:
        assert 0 <= z["p_free"] <= 1 and z["count_source"] == "simulated" and z["confidence"] == "demo"
        assert z["predicted_low"] <= z["predicted_high"]
    assert r["counterfactual_tip"] is None or "22%" not in r["counterfactual_tip"]   # tips are computed, not canned


def test_arrival_time_actually_changes_the_forecast():
    c = client_for_role("student", username="rec_student2", home_block="Block_26_CSE")
    key = lambda r: {z["zone_id"]: z["occupancy_rate"] for z in r["recommended_zones"]}
    soon, later = key(c.get("/api/recommendations?horizon_minutes=5").json()), key(c.get("/api/recommendations?horizon_minutes=120").json())
    assert soon != later                                              # the horizon buttons used to be cosmetic


def test_resolve_arrival_clock_times():
    from api.routes import resolve_arrival
    from fastapi import HTTPException
    now = datetime(2026, 11, 17, 10, 7)
    assert resolve_arrival("17:00", now) == (datetime(2026, 11, 17, 17, 0), 413)          # later today
    assert resolve_arrival("09:00", now) == (datetime(2026, 11, 18, 9, 0), 1373)          # already past: tomorrow
    assert resolve_arrival("2026-11-19T18:00", now)[0] == datetime(2026, 11, 19, 18, 0)
    for bad in ("25:00", "noon", "2026-11-17T09:00", "2026-12-30T09:00"):
        with pytest.raises(HTTPException):
            resolve_arrival(bad, now)


def test_far_arrival_uses_the_calendar_model_and_blends_continuously():
    live.note_real_data()
    HISTORY.record("ZONE_MAINPARK_NORTH", NOW - timedelta(minutes=2), 250)
    by_h = {h: live.predict_all(ZONES_MAP, NOW, h)["ZONE_MAINPARK_NORTH"] for h in (55, 59, 61, 65, 240, 600)}
    assert by_h[55]["forecast_basis"] == "live" and by_h[65]["forecast_basis"] == "blended"
    assert by_h[600]["forecast_basis"] == "calendar"
    assert abs(by_h[61]["predicted_vehicle_count"] - by_h[59]["predicted_vehicle_count"]) <= 6   # no jump at +60
    for p in by_h.values():
        assert p["predicted_low"] <= p["predicted_vehicle_count"] <= p["predicted_high"] and 0 <= p["p_free"] <= 1


def test_far_arrival_follows_time_of_day_not_the_morning():
    """Regression: beyond +60 the old code decayed toward the normal level at reading + 60 min, so 8 PM asked at
    10 AM looked like 11 AM. An academic lot must be far emptier at 8 PM than at 11 AM."""
    live.note_real_data()
    HISTORY.record("ZONE_MAINPARK_NORTH", NOW - timedelta(minutes=2), 290)
    at_11 = live.predict_all(ZONES_MAP, NOW, 53)["ZONE_MAINPARK_NORTH"]["predicted_vehicle_count"]
    at_20 = live.predict_all(ZONES_MAP, NOW, 593)["ZONE_MAINPARK_NORTH"]["predicted_vehicle_count"]
    assert at_20 < at_11 * 0.5


def test_far_arrival_sees_an_exam_day_at_the_arrival_time():
    """A 10 AM forecast made the evening before must know tomorrow is an exam day (exam sessions fill academic
    lots in the morning; the synthetic campus empties them over the exam lunch break, so 1 PM would go the other way)."""
    evening = datetime(2026, 11, 16, 20, 0)
    horizon = 14 * 60                                                   # -> Tue 10:00
    normal = live.predict_all(ZONES_MAP, evening, horizon)["ZONE_MAINPARK_NORTH"]
    calendar_store.add_entry("exam", "Exams", datetime(2026, 11, 17), datetime(2026, 11, 17, 23, 59, 59))
    live.bump()
    with_exam = live.predict_all(ZONES_MAP, evening, horizon)["ZONE_MAINPARK_NORTH"]
    assert with_exam["forecast_basis"] == "calendar"
    assert with_exam["predicted_vehicle_count"] > normal["predicted_vehicle_count"]


def test_recommendations_accept_clock_arrival_times():
    c = client_for_role("student", username="clock_student", home_block="Block_26_CSE")
    r = c.get("/api/recommendations?arrive_at=20:00").json()
    assert r["arrival_time"].endswith("T20:00") and r["horizon_minutes"] >= 1
    assert r["forecast_basis"] in ("live", "blended", "calendar")
    assert all(z["forecast_basis"] in ("live", "blended", "calendar") for z in r["recommended_zones"])
    assert c.get("/api/recommendations?arrive_at=99:00").status_code == 422
    assert 'data-t="17:00"' in c.get("/").text


def test_ranker_prefers_a_ground_that_will_have_space_over_a_closer_one_that_will_not():
    from recommendation.parking_ranker import ParkingRanker
    r = ParkingRanker(seed=0)
    preds = {z: {"predicted_count": int(0.5 * m["operational_capacity"]), "saturation_prob": 0.0,
                 "camera_age_seconds": 10.0, "p_free": 0.95} for z, m in r.zones.items()}
    preds["ZONE_BLOCK1"]["p_free"] = 0.03                             # 30 m from Block 1 but will not have a slot
    assert r.rank_zones("Block_1", preds, apply_anti_herding=False)[0]["zone_id"] != "ZONE_BLOCK1"
    for p in preds.values():                                          # with no forecast at all, distance decides
        p["p_free"] = None
    assert r.rank_zones("Block_1", preds, apply_anti_herding=False)[0]["zone_id"] == "ZONE_BLOCK1"


def test_drivers_already_heading_to_a_ground_reduce_its_free_slot_chance():
    from recommendation.parking_ranker import ParkingRanker
    r = ParkingRanker(seed=0)
    preds = {z: {"predicted_count": int(0.9 * m["operational_capacity"]), "saturation_prob": 0.0,
                 "camera_age_seconds": 10.0, "p_free": 0.8} for z, m in r.zones.items()}
    before = next(z for z in r.rank_zones("Block_1", preds, apply_anti_herding=False) if z["zone_id"] == "ZONE_BLOCK1")["p_free"]
    for _ in range(8):
        r.dispatch_tracker.record_dispatch("ZONE_BLOCK1", max_capacity=140)
    after = next(z for z in r.rank_zones("Block_1", preds, apply_anti_herding=False) if z["zone_id"] == "ZONE_BLOCK1")["p_free"]
    assert after < before


# ─────────────────────────── Ingest -> live loop, monitoring ───────────────────────────

def test_ingested_count_flows_into_snapshot_and_recommendations():
    admin = login_client()
    now = datetime.now().replace(second=0, microsecond=0)
    r = admin.post("/api/ingest", json={"zone_id": "ZONE_LAW", "count": 111, "timestamp": now.isoformat(), "source": "camera"})
    assert r.status_code == 200
    snap = admin.get("/api/snapshot").json()
    law = next(z for z in snap["zones"] if z["zone_id"] == "ZONE_LAW")
    assert law["count_source"] == "camera" and law["current_count"] == 111 and snap["data_mode"] == "live"
    assert snap["zones_by_source"]["camera"] == 1
    rec = admin.get("/api/recommendations?destination=Law_Block&horizon_minutes=15").json()
    assert rec["data_mode"] == "live"
    assert next(z for z in rec["recommended_zones"] if z["zone_id"] == "ZONE_LAW")["count_source"] == "camera"


def test_live_accuracy_scores_logged_forecasts_against_real_counts():
    store = ObservationStore(":memory:")
    base = datetime.now().replace(second=0, microsecond=0)
    base -= timedelta(minutes=base.minute % 5)
    rows = []
    for k in range(60):                                               # 60 forecasts, each 3 vehicles too high, 80% band +-5
        made = base - timedelta(minutes=120 + 5 * k)
        target = made + timedelta(minutes=15)
        store.record("ZONE_LAW", "CAM_LAW_01", 50, target)
        rows.append({"zone_id": "ZONE_LAW", "made_bucket": made.isoformat(), "horizon_min": 15,
                     "target_bucket": target.isoformat(), "predicted": 53.0, "low": 48.0, "high": 58.0,
                     "p_free": 0.9, "capacity": 130, "model_version": "t"})
    model_monitor.log_predictions(rows)
    model_monitor.log_predictions(rows)                               # logging the same forecasts twice must not double count
    acc = model_monitor.live_accuracy(store, 7)
    h = acc["horizons"]["H15"]
    assert h["n"] == 60 and h["MAE"] == 3.0 and h["bias"] == 3.0 and h["interval_coverage"] == 1.0
    assert acc["drift"]["status"] in ("ok", "drift")                  # enough samples to judge
    assert model_monitor.live_accuracy(ObservationStore(":memory:"), 7)["drift"]["status"] == "no_data"


def test_model_status_is_admin_only_and_complete():
    student = client_for_role("student", username="status_student", home_block="Block_1")
    assert student.get("/api/model/status").status_code == 403
    d = login_client().get("/api/model/status").json()
    assert d["loaded"] and d["meta"]["data_source"] == "synthetic" and d["meta"]["version"]
    assert set(d["backtest"]) >= {"H15", "H30", "H60"} and len(d["zones"]) == 12
    assert d["live_accuracy"]["drift"]["status"] in ("no_data", "collecting", "ok", "drift")
    h = login_client().get("/api/health").json()
    assert h["ml_model"]["trained_on"] == "synthetic" and h["data_mode"] in ("demo", "live", "estimated")


def test_model_reload_is_admin_only_and_picks_up_the_loaded_version():
    student = client_for_role("student", username="reload_student", home_block="Block_1")
    assert student.post("/api/model/reload").status_code == 403
    r = login_client().post("/api/model/reload")
    assert r.status_code == 200 and r.json()["status"] == "reloaded" and r.json()["version"] == inference.model_meta()["version"]
    assert login_client().get("/api/audit?action=model_reloaded&limit=1").json()["entries"]


def test_new_pages_render_for_admin_and_are_blocked_for_students():
    admin, student = login_client(), client_for_role("student", username="page_student", home_block="Block_1")
    for page in ("/calendar", "/model"):
        assert admin.get(page).status_code == 200 and student.get(page).status_code == 403
    assert 'href="/calendar"' in admin.get("/admin").text and 'href="/calendar"' not in student.get("/").text
