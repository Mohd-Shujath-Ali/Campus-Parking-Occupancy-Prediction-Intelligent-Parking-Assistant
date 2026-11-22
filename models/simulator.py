"""
Demo feed: plausible live counts for localhost demos when no camera has ever reported.

It replays a normal week of the SYNTHETIC dataset at the matching weekday and time, then layers on
whatever the campus calendar says right now (an admin-entered event adds vehicles around its venue,
an exam day fills the academic lots, a holiday looks like a weekend). Everything it produces is
labelled count_source="simulated" and shown as demo data. It is never used once a real count has
been ingested, so a camera outage cannot silently turn into invented numbers.
"""
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from features import calendar as cal
from features.feature_pipeline import zone_order

DATA_PATH = "data/processed/lpu_timeseries.csv"
_EXAM_CATEGORIES = {"Academic & Administrative Core", "Core Computing, Admission & Innovation", "Core Computing & Labs"}
_BUSY = ((510, 730), (790, 1010))


@lru_cache(maxsize=1)
def _replay():
    """counts[zone_code, weekday, 5-minute slot] from the latest week with no exam, event or holiday."""
    df = pd.read_csv(DATA_PATH, usecols=["timestamp", "zone_id", "vehicle_count", "is_exam_day",
                                         "is_campus_event", "is_holiday"], parse_dates=["timestamp"])
    df["week"] = (df.timestamp - df.timestamp.min().normalize()).dt.days // 7
    flagged = df.groupby("week")[["is_exam_day", "is_campus_event", "is_holiday"]].sum().sum(axis=1)
    full = df.groupby("week").timestamp.nunique() >= 7 * 288
    clean = flagged[(flagged == 0) & full.reindex(flagged.index, fill_value=False)]
    week = int(clean.index.max()) if len(clean) else int(df.week.max())
    w = df[df.week == week]
    order = {z: i for i, z in enumerate(zone_order())}
    counts = np.zeros((len(order), 7, 288))
    z = w.zone_id.map(order).to_numpy()
    d = w.timestamp.dt.dayofweek.to_numpy()
    s = ((w.timestamp.dt.hour * 60 + w.timestamp.dt.minute) // 5).to_numpy()
    counts[z, d, s] = w.vehicle_count.to_numpy()
    return counts


def available() -> bool:
    try:
        _replay()
        return True
    except Exception:
        return False


def _slot(ts: datetime):
    return (ts.hour * 60 + ts.minute) // 5


def counts_at(times: List[datetime], zones: Dict[str, dict]) -> Dict[str, np.ndarray]:
    """Simulated count for each zone at each timestamp, calendar effects included."""
    base = _replay()
    out = {}
    entries = cal.current_entries()
    s = pd.Series(pd.to_datetime(times))
    for zid, meta in zones.items():
        zc = zone_order().index(zid)
        cap = meta["operational_capacity"]
        fl = cal.compute_flags(s, zid, entries)
        dow = s.dt.dayofweek.to_numpy()
        eff_dow = np.where((fl["is_holiday"].to_numpy() > 0) & (dow < 5), 5, dow)     # holiday looks like a Saturday
        slot = ((s.dt.hour * 60 + s.dt.minute) // 5).to_numpy()
        ratio = base[zc, eff_dow, slot] / cap
        mins = (s.dt.hour * 60 + s.dt.minute).to_numpy()
        if meta["category"] in _EXAM_CATEGORIES:
            exam = fl["is_exam_day"].to_numpy() > 0
            busy = np.zeros(len(mins), dtype=bool)
            for a, b in _BUSY:
                busy |= (mins >= a) & (mins <= b)
            ratio = np.where(exam, np.where(busy, np.maximum(ratio, 0.94), ratio * 0.40), ratio)
        ratio = ratio + fl["event_load"].to_numpy()                                      # events add their declared load
        out[zid] = np.round(np.clip(ratio, 0.0, 1.15) * cap).astype(int)
    return out


def history_for(zone_id: str, now: datetime, counts: np.ndarray, times: List[datetime]):
    """A ZoneHistory seeded with the simulated readings strictly before `now`, so lag features are 'observed'."""
    from models.history import ZoneHistory
    h = ZoneHistory()
    for t, c in zip(times, counts):
        if t < now:
            h.record(zone_id, t, int(c))
    return h
