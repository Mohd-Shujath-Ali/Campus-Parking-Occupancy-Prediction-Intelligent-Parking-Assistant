"""
SYNTHETIC LPU parking time series (bootstrap data until real camera counts exist).

Every model trained on this file learns THIS generator, not LPU. It exists so the pipeline,
features, evaluation and UI can be built and tested end to end; artifacts record
data_source="synthetic" and the UI says so. Replace it with real observations as they accumulate
(see models/retrain.py).

The 18-week timeline puts at least one exam window, three events and one holiday in each of the
train / validation / test splits, so event and exam handling is both learnable and testable.
Events realise a random 0.6-1.4x of what the (declared) estimate says, rain days are hidden
shocks the model is never told about, and occupancy noise is autocorrelated like real lots.
"""
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy.signal import lfilter

from features.calendar import (CalendarEntry, compute_flags, day_range, event_ratio_by_zone, zone_ids)

START_DATE = "2026-08-03"      # a Monday
DAYS = 126                     # 18 weeks
SCHEMA_COLUMNS = ["timestamp", "zone_id", "vehicle_count", "operational_capacity", "is_exam_day",
                  "is_campus_event", "is_class_transition", "is_holiday", "event_load",
                  "camera_status", "camera_age_seconds"]

# (day index, first hour, last hour, name, venue block, declared extra vehicles)
_EVENTS = [
    (9, 10, 16, "Orientation Day", "Block_1", 350),
    (26, 8, 18, "Inter-college Sports Meet", "Sports_Ground", 400),
    (44, 9, 17, "Placement Drive", "Block_26_CSE", 250),
    (68, 10, 20, "Tech Fest", "Block_26_CSE", 500),
    (93, 9, 14, "Convocation", "Block_1", 600),
    (101, 12, 21, "Cultural Fest", "MB_Food_Square", 450),
    (105, 10, 15, "Alumni Meet", "Architecture_Block", 200),
    (111, 9, 17, "Sports Day", "Sports_Ground", 350),
    (115, 14, 17, "Guest Lecture Series", "Block_57_Agri", 220),
    (121, 9, 18, "Hackathon", "Block_26_CSE", 450),
]
_EXAMS = [(28, 33, "Mid-term exams"), (91, 96, "End-term exams (set 1)"), (112, 117, "End-term exams (set 2)")]
_HOLIDAYS = [(16, 16, "Festival holiday"), (37, 37, "Local holiday"), (60, 60, "Founders' Day"),
             (98, 98, "National holiday"), (119, 120, "Winter break begins")]
_HIDDEN_RAIN_DAYS = [7, 21, 38, 66, 94, 109]       # +15% occupancy on weekdays; never given to the model
_ACADEMIC = {"Academic & Administrative Core", "Core Computing, Admission & Innovation",
             "Core Computing & Labs", "Design & Applied Arts", "School of Law & Management",
             "Agriculture & Mechanical Core"}
_EXAM_ZONES = {"Academic & Administrative Core", "Core Computing, Admission & Innovation", "Core Computing & Labs"}
_SURGE_CATS = {"Academic & Administrative Core", "Core Computing, Admission & Innovation",
               "Core Computing & Labs", "Design & Applied Arts", "Campus Entrance & Transit"}
_RESIDENTIAL_DINING = {"Residential (Hostels)", "Residential & Sports Arena", "Dining, Hospital & Social Hub"}

_CURVES = {
    "academic": [0.02, 0.01, 0.01, 0.01, 0.02, 0.05, 0.12, 0.35, 0.88, 0.94, 0.91, 0.86,
                 0.72, 0.85, 0.88, 0.82, 0.65, 0.38, 0.18, 0.08, 0.05, 0.03, 0.02, 0.02],
    "dining":   [0.01, 0.01, 0.01, 0.01, 0.02, 0.04, 0.08, 0.20, 0.45, 0.55, 0.50, 0.65,
                 0.92, 0.98, 0.85, 0.60, 0.68, 0.85, 0.90, 0.75, 0.45, 0.25, 0.10, 0.04],
    "hostel":   [0.85, 0.88, 0.89, 0.90, 0.88, 0.80, 0.65, 0.45, 0.30, 0.25, 0.28, 0.30,
                 0.35, 0.40, 0.38, 0.35, 0.45, 0.65, 0.78, 0.85, 0.88, 0.88, 0.86, 0.85],
    "transit":  [0.05, 0.03, 0.02, 0.02, 0.05, 0.12, 0.28, 0.55, 0.92, 0.90, 0.85, 0.82,
                 0.80, 0.84, 0.82, 0.80, 0.82, 0.75, 0.55, 0.35, 0.20, 0.12, 0.08, 0.06],
    "sports":   [0.01, 0.01, 0.01, 0.01, 0.02, 0.15, 0.35, 0.20, 0.15, 0.15, 0.18, 0.20,
                 0.25, 0.22, 0.20, 0.35, 0.65, 0.85, 0.88, 0.60, 0.25, 0.10, 0.04, 0.02],
}
_CATEGORY_CURVE = {
    "Campus Entrance & Transit": "transit", "Dining, Hospital & Social Hub": "dining",
    "Dining & Hostel Block 41-D": "dining", "Residential (Hostels)": "hostel",
    "Residential & Sports Arena": "hostel", "Athletics & Mechanical Link": "sports",
}


def synthetic_calendar(start_date: str = START_DATE, days: int = DAYS) -> List[CalendarEntry]:
    """The exam windows, holidays and events baked into the synthetic data."""
    d0 = pd.Timestamp(start_date).date()
    out = [day_range(d0 + timedelta(days=a), d0 + timedelta(days=b), "exam", n) for a, b, n in _EXAMS if b < days]
    out += [day_range(d0 + timedelta(days=a), d0 + timedelta(days=b), "holiday", n) for a, b, n in _HOLIDAYS if b < days]
    for day, h0, h1, name, venue, veh in _EVENTS:
        if day < days:
            base = datetime.combine(d0 + timedelta(days=day), datetime.min.time())
            out.append(CalendarEntry("event", name, base + timedelta(hours=h0), base + timedelta(hours=h1),
                                     venue_block=venue, expected_vehicles=veh))
    return out


def generate_lpu_parking_dataset(
    config_path: str = "configs/zones.yaml",
    output_csv_path: str = "data/processed/lpu_timeseries.csv",
    output_parquet_path: str = "data/processed/lpu_timeseries.parquet",
    start_date: str = START_DATE,
    days: int = DAYS,
    cadence_minutes: int = 5,
    seed: int = 42,
    write: bool = True,
) -> pd.DataFrame:
    with open(config_path, "r") as f:
        zones = sorted(yaml.safe_load(f)["zones"], key=lambda z: z["id"])
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start=start_date, periods=int(days * 24 * 60 / cadence_minutes), freq=f"{cadence_minutes}min")
    n = len(idx)
    entries = synthetic_calendar(start_date, days)
    events = [e for e in entries if e.kind == "event"]
    realised = {e.name: float(rng.uniform(0.6, 1.4)) for e in events}   # what actually turned up vs the estimate
    day_no = np.asarray((idx - pd.Timestamp(start_date)).days)
    hr = np.asarray(idx.hour + idx.minute / 60.0)
    mins = np.asarray(idx.hour * 60 + idx.minute)
    rain = np.isin(day_no, _HIDDEN_RAIN_DAYS)
    d0 = pd.Timestamp(start_date)
    outage = (idx >= d0 + timedelta(days=14, hours=14)) & (idx <= d0 + timedelta(days=14, hours=15, minutes=30))

    frames = []
    for zi, zone in enumerate(zones):
        zid, cap, cat = zone["id"], zone["operational_capacity"], zone["category"]
        curve = np.array(_CURVES[_CATEGORY_CURVE.get(cat, "academic")])
        flags = compute_flags(idx, zid, entries)
        weekend = np.asarray(idx.dayofweek >= 5)
        off_day = weekend | (flags["is_holiday"].to_numpy() > 0)
        school = ~off_day

        base = np.interp(hr, np.arange(25), np.append(curve, curve[0]))
        base = np.where(off_day, np.minimum(1.0, base * 1.15) if cat in _RESIDENTIAL_DINING else base * 0.18, base)

        surge = np.zeros(n)
        if cat in _SURGE_CATS:
            surge = np.where(school & (mins >= 525) & (mins <= 555), 0.14, surge)
            surge = np.where(school & (mins >= 645) & (mins <= 675), 0.08, surge)
            surge = np.where(school & (mins >= 765) & (mins <= 810), -0.10, surge)
            surge = np.where(school & (mins >= 1005) & (mins <= 1050), -0.18, surge)

        if cat in _EXAM_ZONES:
            exam = flags["is_exam_day"].to_numpy() > 0
            busy = ((mins >= 510) & (mins <= 730)) | ((mins >= 790) & (mins <= 1010))
            base = np.where(exam, np.where(busy, np.maximum(base, 0.94), base * 0.40), base)

        for e in events:                                    # realised demand: ramps up/down over 30 minutes
            on = ((idx >= e.start) & (idx <= e.end))
            if on.any():
                since = np.asarray((idx - e.start).total_seconds())
                until = np.asarray((e.end - idx).total_seconds())
                ramp = np.clip(np.minimum(since, until) / 1800.0, 0, 1)
                base = base + on * ramp * realised[e.name] * event_ratio_by_zone(e)[zi]

        base = np.where(rain & school, np.minimum(1.08, base + 0.15), base)

        # AR(1) occupancy noise (sd 3%, persistent) + white detector noise (sd 1.2%)
        phi, sd = 0.92, 0.03
        ar = lfilter([1.0], [1.0, -phi], rng.normal(0, sd * np.sqrt(1 - phi ** 2), n))
        ratio = np.clip(base + surge + ar + rng.normal(0, 0.012, n), 0.0, 1.15)

        cam_off = outage if zid == "ZONE_BLOCK1" else np.zeros(n, dtype=bool)
        frames.append(pd.DataFrame({
            "timestamp": idx, "zone_id": zid, "vehicle_count": np.round(ratio * cap).astype(int),
            "operational_capacity": cap,
            "is_exam_day": flags["is_exam_day"], "is_campus_event": flags["is_campus_event"],
            "is_class_transition": flags["is_class_transition"], "is_holiday": flags["is_holiday"],
            "event_load": flags["event_load"].round(4),
            "camera_status": np.where(cam_off, "offline", "live"),
            "camera_age_seconds": np.where(cam_off, (idx - (d0 + timedelta(days=14, hours=14))).total_seconds(),
                                           rng.integers(5, 45, n).astype(float)),
        }))

    full = pd.concat(frames, ignore_index=True).sort_values(["timestamp", "zone_id"]).reset_index(drop=True)
    full = full[SCHEMA_COLUMNS]
    if write:
        Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
        full.to_csv(output_csv_path, index=False)
        print(f"Generated {len(full)} records across {len(zones)} zones -> {output_csv_path}")
        try:
            full.to_parquet(output_parquet_path, index=False)
            print(f"Saved Parquet format -> {output_parquet_path}")
        except Exception as e:                                   # parquet engine is optional
            Path(output_parquet_path).unlink(missing_ok=True)    # never leave a stale copy next to the new CSV
            print(f"Parquet export skipped (engine missing); removed any stale copy. {type(e).__name__}")
        cal_path = Path("data/external/synthetic_calendar.json")
        cal_path.parent.mkdir(parents=True, exist_ok=True)
        cal_path.write_text(json.dumps([
            {"kind": e.kind, "name": e.name, "start": e.start.isoformat(), "end": e.end.isoformat(),
             "venue_block": e.venue_block, "expected_vehicles": e.expected_vehicles,
             "realised_factor": realised.get(e.name)} for e in entries], indent=2))
    return full


if __name__ == "__main__":
    generate_lpu_parking_dataset()
