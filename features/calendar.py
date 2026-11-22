"""
Campus calendar: bells, exam windows, holidays and admin-entered events.

Everything the forecaster knows in advance about a time lives here, so training, the synthetic
data generator and live serving all derive the same flags from the same rules:

* is_class_transition : weekday, within +-15 min of a lecture bell, not a holiday
* is_exam_day         : inside an exam window, on a weekday
* is_holiday          : inside a holiday range (the lots behave like a weekend)
* is_campus_event     : an event is on
* event_load          : expected extra occupancy RATIO this event adds to a zone, from the admin's
                        estimate of vehicles, spread over grounds by walking distance from the venue

Events are rare, so the model is given the admin's declared estimate (event_load) as a feature and
learns how much to trust it, rather than having to infer events from a binary flag.
"""
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from functools import lru_cache
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from utils.config import load_model_config, load_zones_config

KINDS = ("exam", "holiday", "event")
EVENT_DECAY_M = 250.0          # metres: how quickly an event's demand falls off around its venue
MAX_EVENT_LOAD = 0.60          # cap on the extra occupancy ratio one event can add to one ground
BELL_HALF_WINDOW_MIN = 15
MINS_TO_BELL_CAP = 60


@dataclass(frozen=True)
class CalendarEntry:
    kind: str                           # exam | holiday | event
    name: str
    start: datetime                     # naive campus-local (IST) time
    end: datetime
    venue_block: Optional[str] = None   # events only: a block id from configs/campus_blocks.yaml
    expected_vehicles: int = 0          # events only: admin's estimate of extra vehicles
    id: Optional[int] = None

    def covers_day(self, d: date) -> bool:
        return self.start.date() <= d <= self.end.date()


def day_range(start: date, end: date, kind: str, name: str) -> CalendarEntry:
    """Whole-day entry (exam window / holiday), inclusive of both dates."""
    return CalendarEntry(kind, name, datetime.combine(start, datetime.min.time()),
                         datetime.combine(end, datetime.max.time().replace(microsecond=0)))


# --------------------------------------------------------------------------- provider
_provider: Callable[[], List[CalendarEntry]] = lambda: []


def set_provider(fn: Callable[[], List[CalendarEntry]]) -> None:
    """Serving registers the database-backed source; training and tests use an explicit list."""
    global _provider
    _provider = fn


def current_entries() -> List[CalendarEntry]:
    return list(_provider())


# --------------------------------------------------------------------------- zone layout
@lru_cache(maxsize=1)
def _zones():
    zs = sorted(load_zones_config()["zones"], key=lambda z: z["id"])   # same order as zone_code()
    return [z["id"] for z in zs], np.array([z["operational_capacity"] for z in zs], dtype=float), zs


def zone_ids() -> List[str]:
    return _zones()[0]


@lru_cache(maxsize=64)
def event_zone_weights(venue_block: str) -> np.ndarray:
    """Share of an event's vehicles each ground takes (sums to 1), by walking distance from the venue."""
    from recommendation.campus_blocks import get_campus_blocks
    anchor = get_campus_blocks().resolve_anchor(venue_block)
    if anchor is None:
        raise ValueError(f"Unknown venue block '{venue_block}'.")
    _, _, zs = _zones()
    d = np.array([float(z["walking_distances_meters"][anchor]) for z in zs])
    w = np.exp(-d / EVENT_DECAY_M)
    return w / w.sum()


def event_ratio_by_zone(entry: CalendarEntry) -> np.ndarray:
    """Declared extra occupancy ratio per zone (indexed by zone_code) for an event entry."""
    _, caps, _ = _zones()
    extra = entry.expected_vehicles * event_zone_weights(entry.venue_block)
    return np.minimum(extra / caps, MAX_EVENT_LOAD)


def event_impact(entry: CalendarEntry, top: int = 4) -> List[Dict]:
    """Human-readable preview of which grounds an event loads, for the admin UI."""
    ids, caps, zs = _zones()
    ratio = event_ratio_by_zone(entry)
    extra = entry.expected_vehicles * event_zone_weights(entry.venue_block)
    order = np.argsort(-ratio)[:top]
    return [{"zone_id": ids[i], "name": zs[i]["name"], "extra_vehicles": int(round(extra[i])),
             "extra_occupancy_pct": int(round(100 * ratio[i]))} for i in order]


# --------------------------------------------------------------------------- bells
@lru_cache(maxsize=1)
def bells() -> np.ndarray:
    return np.array(load_model_config()["features"]["bell_times_minutes_of_day"], dtype=float)


def _as_series(ts) -> pd.Series:
    return pd.Series(pd.to_datetime(ts)).reset_index(drop=True)


def mins_to_bell(ts, no_class: np.ndarray) -> np.ndarray:
    """Minutes to the nearest lecture bell (capped); the cap on days without classes."""
    s = _as_series(ts)
    mins = (s.dt.hour * 60 + s.dt.minute).to_numpy(dtype=float)
    d = np.abs(mins[:, None] - bells()[None, :]).min(axis=1)
    return np.where(no_class, MINS_TO_BELL_CAP, np.minimum(d, MINS_TO_BELL_CAP))


def class_transition(ts, is_holiday) -> np.ndarray:
    s = _as_series(ts)
    weekday = (s.dt.dayofweek < 5).to_numpy()
    no_class = ~weekday | (np.asarray(is_holiday) > 0)
    return ((mins_to_bell(s, no_class) <= BELL_HALF_WINDOW_MIN) & ~no_class).astype(int)


# --------------------------------------------------------------------------- flags
def compute_flags(ts, zones: Union[str, Sequence[str]], entries: Optional[Iterable[CalendarEntry]] = None
                  ) -> pd.DataFrame:
    """
    Calendar flags for each (timestamp, zone) row. `zones` is one id for all rows or one per row.
    Returns columns is_exam_day, is_holiday, is_campus_event, event_load, is_class_transition,
    mins_to_bell (all aligned with `ts`).
    """
    entries = current_entries() if entries is None else list(entries)
    s = _as_series(ts)
    n = len(s)
    ids = zone_ids()
    if isinstance(zones, str):
        zcode = np.full(n, ids.index(zones) if zones in ids else -1)
    else:
        lookup = {z: i for i, z in enumerate(ids)}
        zcode = np.array([lookup.get(z, -1) for z in zones])
    valid = zcode >= 0
    day = s.dt.normalize()
    weekday = (s.dt.dayofweek < 5).to_numpy()

    exam = np.zeros(n, dtype=bool)
    holiday = np.zeros(n, dtype=bool)
    event = np.zeros(n, dtype=bool)
    load = np.zeros(n)
    for e in entries:
        if e.kind == "exam":
            exam |= ((day >= pd.Timestamp(e.start.date())) & (day <= pd.Timestamp(e.end.date()))).to_numpy()
        elif e.kind == "holiday":
            holiday |= ((day >= pd.Timestamp(e.start.date())) & (day <= pd.Timestamp(e.end.date()))).to_numpy()
        elif e.kind == "event":
            on = ((s >= pd.Timestamp(e.start)) & (s <= pd.Timestamp(e.end))).to_numpy()
            event |= on
            if on.any():
                r = event_ratio_by_zone(e)
                load += np.where(on & valid, r[np.where(valid, zcode, 0)], 0.0)
    exam &= weekday & ~holiday
    no_class = ~weekday | holiday
    near = mins_to_bell(s, no_class)
    return pd.DataFrame({
        "is_exam_day": exam.astype(int),
        "is_holiday": holiday.astype(int),
        "is_campus_event": event.astype(int),
        "event_load": np.minimum(load, 1.0),
        "is_class_transition": ((near <= BELL_HALF_WINDOW_MIN) & ~no_class).astype(int),
        "mins_to_bell": near,
    })


def flags_at(ts: datetime, zone_id: str, entries: Optional[Iterable[CalendarEntry]] = None) -> Dict[str, float]:
    """Flags for one zone at one instant."""
    return {k: float(v) for k, v in compute_flags([ts], zone_id, entries).iloc[0].items()}


def context_at(ts: datetime, entries: Optional[Iterable[CalendarEntry]] = None) -> Dict:
    """What is special about this moment campus-wide (shown to drivers and admins)."""
    entries = current_entries() if entries is None else list(entries)
    f = compute_flags([ts], zone_ids()[0], [e for e in entries if e.kind != "event"]).iloc[0]
    active = [e for e in entries if e.kind == "event" and e.start <= ts <= e.end]
    upcoming = [e for e in entries if e.kind == "event" and ts < e.start <= ts + timedelta(hours=3)]
    names = {k: [e.name for e in entries if e.kind == k and e.covers_day(ts.date())] for k in ("exam", "holiday")}
    return {
        "bell_window": bool(f["is_class_transition"]),
        "exam_day": bool(f["is_exam_day"]), "exam_names": names["exam"],
        "holiday": bool(f["is_holiday"]), "holiday_names": names["holiday"],
        "weekend": ts.weekday() >= 5,
        "events_now": [{"name": e.name, "venue_block": e.venue_block, "ends": e.end.isoformat()} for e in active],
        "events_soon": [{"name": e.name, "venue_block": e.venue_block, "starts": e.start.isoformat()} for e in upcoming],
    }
