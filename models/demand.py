"""
Parking Demand Forecasting (core module 5): days-ahead occupancy profile per zone.

Unlike the short-horizon forecaster (which starts from the live count and lags), this model
uses CALENDAR features only (zone, time of day, weekday, exam/event flags), so it can look
7+ days ahead and answer "what if it is an exam week?" questions for administrators.
It predicts the occupancy RATIO (vehicles / capacity) so zones of different size share one model.
"""
from datetime import date, datetime, timedelta
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from features.feature_pipeline import zone_code, zone_order
from utils.config import load_zones_config, xgb_params

CAL_COLS = ["zone_code", "hour_sin", "hour_cos", "day_sin", "day_cos",
            "is_weekend", "is_exam_day", "is_campus_event"]


def calendar_frame(ts: pd.Series, zone_ids: pd.Series, exam: Iterable[int] = 0,
                   event: Iterable[int] = 0) -> pd.DataFrame:
    """Calendar-only design matrix for the given timestamps/zones."""
    ts = pd.to_datetime(pd.Series(ts).reset_index(drop=True))
    hour = ts.dt.hour + ts.dt.minute / 60.0
    dow = ts.dt.dayofweek
    return pd.DataFrame({
        "zone_code": [zone_code(z) for z in pd.Series(zone_ids).reset_index(drop=True)],
        "hour_sin": np.sin(2 * np.pi * hour / 24), "hour_cos": np.cos(2 * np.pi * hour / 24),
        "day_sin": np.sin(2 * np.pi * dow / 7), "day_cos": np.cos(2 * np.pi * dow / 7),
        "is_weekend": (dow >= 5).astype(int),
        "is_exam_day": np.asarray(exam) * np.ones(len(ts), dtype=int),
        "is_campus_event": np.asarray(event) * np.ones(len(ts), dtype=int),
    })[CAL_COLS]


def frame_from_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar features of a raw observation frame (with the flag columns)."""
    return calendar_frame(df["timestamp"], df["zone_id"], df["is_exam_day"].values,
                          df["is_campus_event"].values)


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

    # ------------------------------------------------------------------ forecasting
    def forecast_grid(self, start: date, days: int = 7, step_min: int = 30,
                      exam_dates: Iterable[date] = (), event_dates: Iterable[date] = (),
                      zone_ids: Optional[List[str]] = None) -> pd.DataFrame:
        zones = zone_ids or list(self.caps)
        exam, event = set(exam_dates), set(event_dates)
        idx = pd.date_range(datetime.combine(start, datetime.min.time()),
                            periods=int(days * 24 * 60 / step_min), freq=f"{step_min}min")
        rows = []
        for z in zones:
            d = calendar_frame(pd.Series(idx), pd.Series([z] * len(idx)),
                               [int(t.date() in exam) for t in idx], [int(t.date() in event) for t in idx])
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
