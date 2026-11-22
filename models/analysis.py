"""
Zone-Level Analysis and Heatmaps (core modules 6 and 9).

Computed from the historical occupancy table (data/processed/lpu_timeseries.csv, currently
SIMULATED; swap the path for an export of real observations once collected).
"""
import os
import logging
from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from utils.config import load_zones_config

logger = logging.getLogger(__name__)

HIST_PATH = "data/processed/lpu_timeseries.csv"
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


@lru_cache(maxsize=1)
def history() -> pd.DataFrame:
    if not os.path.exists(HIST_PATH):
        logger.warning("Historical timeseries (%s) not found. Running in lean mode without historical analytics.", HIST_PATH)
        return pd.DataFrame(columns=["timestamp", "zone_id", "vehicle_count", "operational_capacity", "occ", "hour", "dow", "weekend", "date"])
    df = pd.read_csv(HIST_PATH, usecols=["timestamp", "zone_id", "vehicle_count", "operational_capacity"],
                     parse_dates=["timestamp"])
    df["occ"] = df.vehicle_count / df.operational_capacity
    df["hour"] = df.timestamp.dt.hour
    df["dow"] = df.timestamp.dt.dayofweek
    df["weekend"] = df.dow >= 5
    df["date"] = df.timestamp.dt.date
    return df


def _hourly(df: pd.DataFrame) -> List[float]:
    h = df.groupby("hour").occ.mean()
    return [round(float(h.get(i, 0.0)), 3) for i in range(24)]


def zone_analysis(zone_id: str) -> Dict:
    h = history()
    z = h[h.zone_id == zone_id]
    if z.empty:
        raise KeyError(zone_id)
    # NOTE: vehicle turnover is deliberately not reported. Counts only give net change per
    # interval (and 5-minute counts are noisy), so a churn figure would mostly measure noise.
    wd, we = z[~z.weekend], z[z.weekend]
    prof_wd = pd.Series(_hourly(wd))
    peak_h = int(prof_wd.idxmax())

    # first hour of a weekday at which the lot passes 75% / falls back under 50% after the peak
    def first_cross(day: pd.DataFrame, above: bool):
        hh = day.groupby("hour").occ.mean()
        if above:
            hit = hh[hh >= 0.75]
            return int(hit.index.min()) if len(hit) else None
        after = hh[(hh.index > peak_h) & (hh < 0.5)]
        return int(after.index.min()) if len(after) else None

    per_day = [(first_cross(g, True), first_cross(g, False)) for _, g in wd.groupby("date")]
    fills = [a for a, _ in per_day if a is not None]
    drains = [b for _, b in per_day if b is not None]

    return {
        "zone_id": zone_id,
        "capacity": int(z.operational_capacity.iloc[0]),
        "data_source": "simulated",
        "mean_occupancy": round(float(z.occ.mean()), 3),
        "p95_occupancy": round(float(z.occ.quantile(0.95)), 3),
        "share_time_saturated": round(float((z.occ >= 0.9).mean()), 3),
        "weekday_profile": _hourly(wd),
        "weekend_profile": _hourly(we),
        "weekday_peak_hour": peak_h,
        "weekday_peak_occupancy": round(float(prof_wd.max()), 3),
        "typical_fill_hour": int(np.median(fills)) if fills else None,   # first >=75%
        "typical_drain_hour": int(np.median(drains)) if drains else None,  # back under 50%
        "weekend_to_weekday_ratio": round(float(we.occ.mean() / max(wd.occ.mean(), 1e-9)), 2),
    }


def heatmap(kind: str, zone_id: Optional[str] = None, day_type: str = "weekday") -> Dict:
    h = history()
    if kind == "zone_hour":
        if h.empty:
            names = {z["id"]: z["name"] for z in load_zones_config()["zones"]}
            order = list(names.keys())
            return {"kind": kind, "day_type": day_type, "rows": [names[z] for z in order],
                    "row_ids": order, "cols": list(range(24)),
                    "values": [[0.0] * 24 for _ in order], "data_source": "none"}
        d = h[h.weekend] if day_type == "weekend" else h[~h.weekend]
        pv = d.pivot_table(index="zone_id", columns="hour", values="occ", aggfunc="mean")
        pv = pv.reindex(columns=range(24)).fillna(0)
        order = pv.mean(axis=1).sort_values(ascending=False).index
        names = {z["id"]: z["name"] for z in load_zones_config()["zones"]}
        return {"kind": kind, "day_type": day_type, "rows": [names.get(z, z) for z in order],
                "row_ids": list(order), "cols": list(range(24)),
                "values": pv.loc[order].round(3).values.tolist(), "data_source": "simulated"}
    if kind == "zone_week":
        if not zone_id:
            raise ValueError("zone_id is required for kind=zone_week")
        d = h[h.zone_id == zone_id]
        if d.empty:
            raise KeyError(zone_id)
        pv = d.pivot_table(index="dow", columns="hour", values="occ", aggfunc="mean").reindex(
            index=range(7), columns=range(24)).fillna(0)
        return {"kind": kind, "zone_id": zone_id, "rows": DAYS, "cols": list(range(24)),
                "values": pv.round(3).values.tolist(), "data_source": "simulated"}
    raise ValueError("kind must be 'zone_hour' or 'zone_week'")
