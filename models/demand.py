"""
Parking Demand Forecasting (core module 5): days-ahead occupancy profile per zone.

Unlike the short-horizon forecaster (which starts from the live count and lags), this model
uses CALENDAR features only (zone, time of day, weekday, exam/holiday flags and the declared load of
events), so it can look 7+ days ahead and answer "what if it is an exam week?" questions for
administrators. It predicts the occupancy RATIO (vehicles / capacity) so zones of different size
share one model.
"""
from datetime import date, datetime, timedelta
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from features import calendar as cal
from features.feature_pipeline import zone_code, zone_order
from utils.config import load_zones_config, xgb_params

CAL_COLS = ["zone_code", "hour_sin", "hour_cos", "day_sin", "day_cos",
            "is_weekend", "is_holiday", "is_exam_day", "is_campus_event", "event_load"]

# A what-if "campus event on this date" (no venue given) is modelled as this much extra occupancy everywhere.
GENERIC_EVENT_LOAD = 0.15


def calendar_frame(ts: pd.Series, zone_ids: pd.Series, exam: Iterable[int] = 0, event: Iterable[int] = 0,
                   holiday: Iterable[int] = 0, event_load: Iterable[float] = 0.0) -> pd.DataFrame:
    """Calendar-only design matrix for the given timestamps/zones."""
    ts = pd.to_datetime(pd.Series(ts).reset_index(drop=True))
    hour = ts.dt.hour + ts.dt.minute / 60.0
    dow = ts.dt.dayofweek
    n = len(ts)
    one = np.ones(n)
    return pd.DataFrame({
        "zone_code": [zone_code(z) for z in pd.Series(zone_ids).reset_index(drop=True)],
        "hour_sin": np.sin(2 * np.pi * hour / 24), "hour_cos": np.cos(2 * np.pi * hour / 24),
        "day_sin": np.sin(2 * np.pi * dow / 7), "day_cos": np.cos(2 * np.pi * dow / 7),
        "is_weekend": (dow >= 5).astype(int),
        "is_holiday": np.asarray(holiday) * np.ones(n, dtype=int),
        "is_exam_day": np.asarray(exam) * np.ones(n, dtype=int),
        "is_campus_event": np.asarray(event) * np.ones(n, dtype=int),
        "event_load": np.asarray(event_load, dtype=float) * one,
    })[CAL_COLS]


def frame_from_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar features of a raw observation frame (with the flag columns)."""
    zeros = np.zeros(len(df))
    return calendar_frame(df["timestamp"], df["zone_id"], df["is_exam_day"].values,
                          df["is_campus_event"].values,
                          df["is_holiday"].values if "is_holiday" in df else zeros,
                          df["event_load"].values if "event_load" in df else zeros)


class DemandForecaster:
    def __init__(self):
        p = xgb_params("xgboost_regressor", tuned=False)
        p.update(n_estimators=300, max_depth=7, learning_rate=0.05, min_child_weight=5)
        pre = ColumnTransformer([
            ("zone", OneHotEncoder(categories=[list(range(len(zone_order())))],
                                   handle_unknown="ignore", sparse_output=False), ["zone_code"]),
            ("rest", "passthrough", CAL_COLS[1:]),
        ])
        self.pipe = Pipeline([("prep", pre), ("model", xgb.XGBRegressor(**p))])
        self.caps = {z["id"]: z["operational_capacity"] for z in load_zones_config()["zones"]}

    def fit(self, df_raw: pd.DataFrame):
        ratio = df_raw["vehicle_count"] / df_raw["operational_capacity"]
        self.pipe.fit(frame_from_raw(df_raw), ratio.values)
        return self

    def predict_ratio(self, X_cal: pd.DataFrame) -> np.ndarray:
        return np.clip(self.pipe.predict(X_cal), 0.0, 1.2)

    def predict_at(self, zone_ids: List[str], times: List[datetime],
                   entries: Optional[Iterable[cal.CalendarEntry]] = None):
        """Expected occupancy ratio for each (zone, time) pair, with the campus calendar applied.
        Returns (ratio, event_flag) arrays aligned with the inputs."""
        ts = pd.Series(pd.to_datetime(list(times)))
        fl = cal.compute_flags(ts, list(zone_ids), entries)
        X = calendar_frame(ts, pd.Series(list(zone_ids)), fl["is_exam_day"].to_numpy(), fl["is_campus_event"].to_numpy(),
                           fl["is_holiday"].to_numpy(), fl["event_load"].to_numpy())
        return self.predict_ratio(X), (fl["event_load"].to_numpy() > 0).astype(int)

    # ------------------------------------------------------------------ forecasting
    def forecast_grid(self, start: date, days: int = 7, step_min: int = 30,
                      exam_dates: Iterable[date] = (), event_dates: Iterable[date] = (),
                      zone_ids: Optional[List[str]] = None, holiday_dates: Iterable[date] = (),
                      entries: Optional[Iterable[cal.CalendarEntry]] = None) -> pd.DataFrame:
        """
        Expected occupancy per zone every `step_min` minutes. `entries` are the campus calendar entries
        (exams, holidays, events with venue and expected vehicles); the *_dates arguments add what-if
        days on top (a what-if event day uses GENERIC_EVENT_LOAD everywhere).
        """
        zones = zone_ids or list(self.caps)
        exam, event, hol = set(exam_dates), set(event_dates), set(holiday_dates)
        entries = list(entries or [])
        idx = pd.date_range(datetime.combine(start, datetime.min.time()),
                            periods=int(days * 24 * 60 / step_min), freq=f"{step_min}min")
        dates = pd.Series(idx).dt.date
        rows = []
        for z in zones:
            fl = cal.compute_flags(pd.Series(idx), z, entries)
            what_if_exam = dates.isin(exam).to_numpy().astype(int)
            what_if_event = dates.isin(event).to_numpy().astype(int)
            what_if_hol = dates.isin(hol).to_numpy().astype(int)
            weekday = (pd.Series(idx).dt.dayofweek < 5).to_numpy()
            holiday = np.maximum(fl["is_holiday"].to_numpy(), what_if_hol)
            examf = np.maximum(fl["is_exam_day"].to_numpy(), what_if_exam * weekday * (1 - holiday))
            d = calendar_frame(pd.Series(idx), pd.Series([z] * len(idx)), examf,
                               np.maximum(fl["is_campus_event"].to_numpy(), what_if_event), holiday,
                               np.minimum(fl["event_load"].to_numpy() + what_if_event * GENERIC_EVENT_LOAD, 1.0))
            ratio = self.predict_ratio(d)
            rows.append(pd.DataFrame({"timestamp": idx, "zone_id": z, "occupancy_ratio": ratio,
                                      "capacity": self.caps[z]}))
        out = pd.concat(rows, ignore_index=True)
        out["vehicles"] = (out.occupancy_ratio * out.capacity).round()
        return out


def summarise_days(grid: pd.DataFrame, step_min: int = 30) -> List[Dict]:
    """Per zone and day: peak time/level, time spent saturated, when the lot first gets busy."""
    out = []
    g = grid.assign(date=grid.timestamp.dt.date)
    for (z, d), part in g.groupby(["zone_id", "date"], sort=True):
        peak = part.loc[part.occupancy_ratio.idxmax()]
        above75 = part[part.occupancy_ratio >= 0.75]
        hourly = part.groupby(part.timestamp.dt.hour).occupancy_ratio.mean().round(3)
        out.append({
            "zone_id": z, "date": d.isoformat(), "weekday": pd.Timestamp(d).day_name(),
            "peak_time": peak.timestamp.strftime("%H:%M"),
            "peak_occupancy": round(float(peak.occupancy_ratio), 3),
            "peak_vehicles": int(peak.vehicles),
            "hours_above_90": round(float((part.occupancy_ratio >= 0.9).sum() * step_min / 60), 1),
            "first_time_above_75": above75.timestamp.min().strftime("%H:%M") if len(above75) else None,
            "hourly": [float(hourly.get(h, 0.0)) for h in range(24)],
        })
    return out
