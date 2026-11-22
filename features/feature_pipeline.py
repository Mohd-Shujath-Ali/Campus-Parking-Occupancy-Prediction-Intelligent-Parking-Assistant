import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
from functools import lru_cache

from features.calendar import mins_to_bell
from features.profile import ProfileModel
from utils.config import load_zones_config

# Forecast horizons (minutes) the direct models are trained for.
HORIZONS: Tuple[int, ...] = (15, 30, 60)

# Single definition of the model input schema, shared by training and serving
# so the two can never drift apart.
#
# State at time t: the live count, recent history, the calendar flags for t and how far today
# deviates from a normal day (profile_dev).
BASE_COLS: List[str] = [
    "vehicle_count", "operational_capacity", "occupancy_ratio",
    "zone_code",
    "hour_sin", "hour_cos", "day_sin", "day_cos", "is_weekend", "is_holiday",
    "is_exam_day", "is_campus_event", "event_load", "is_class_transition",
    "lag_15m", "lag_30m", "lag_45m", "lag_60m",
    "delta_15m", "delta_30m",
    "rolling_mean_30m", "rolling_std_30m",
    "rolling_mean_60m", "rolling_std_60m",
    "profile_now", "profile_dev",
]
# What is already known about the ARRIVAL time t+h (calendar and timetable are known in advance):
# the normal-day profile there, events, exams, holidays, and distance to a lecture bell.
HORIZON_FIELDS: Tuple[str, ...] = ("profile", "event_load", "is_exam_day", "is_holiday", "mins_to_bell")
HORIZON_COLS: List[str] = [f"tgt{h}_{f}" for h in HORIZONS for f in HORIZON_FIELDS]
FEATURE_COLS: List[str] = BASE_COLS + HORIZON_COLS


def horizon_col(h: int, field: str) -> str:
    return f"tgt{h}_{field}"


@lru_cache(maxsize=1)
def zone_order() -> Tuple[str, ...]:
    """Stable zone ordering (sorted ids from zones.yaml)."""
    return tuple(sorted(z["id"] for z in load_zones_config()["zones"]))


def zone_code(zone_id: str) -> int:
    """Integer code for a zone; -1 for an unknown zone."""
    order = zone_order()
    return order.index(zone_id) if zone_id in order else -1


class TimeSeriesFeaturePipeline:
    """
    Constructs causal time-series features (cyclics, lags, velocity, rolling windows, calendar and
    profile features) without future data leakage.

    All shift()/rolling() operations are applied per zone, on a complete 5-minute grid, so lag
    values never bleed across zones and a gap (camera outage) yields missing lags and targets
    instead of silently pairing the wrong rows.

    `profile`: a fitted ProfileModel. Without one, profile features fall back to the current
    occupancy (i.e. "no information beyond persistence"), which keeps the schema valid for unit
    tests but is not what production training uses. `profile_leave_week_out=True` is for the TRAIN
    split only: each row then sees a profile that excludes its own week.
    """

    def __init__(
        self,
        lag_minutes: List[int] = [15, 30, 45, 60],
        rolling_windows_minutes: List[int] = [30, 60],
        saturation_threshold: float = 0.90,
        profile: Optional[ProfileModel] = None,
        profile_leave_week_out: bool = False,
    ):
        self.saturation_threshold = saturation_threshold
        self.lag_minutes = lag_minutes
        self.rolling_windows_minutes = rolling_windows_minutes
        self.profile = profile
        self.profile_leave_week_out = profile_leave_week_out

    # ------------------------------------------------------------------
    # Internal per-zone transform
    # ------------------------------------------------------------------

    @staticmethod
    def _complete_grid(df: pd.DataFrame, cadence_minutes: int) -> pd.DataFrame:
        full = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq=f"{cadence_minutes}min")
        if len(full) == len(df):
            return df
        out = df.set_index("timestamp").reindex(full).rename_axis("timestamp").reset_index()
        for c in ("zone_id", "operational_capacity"):
            if c in out:
                out[c] = out[c].ffill().bfill()
        return out

    def _transform_single_zone(self, df: pd.DataFrame, cadence_minutes: int) -> pd.DataFrame:
        df = df.copy().sort_values("timestamp").reset_index(drop=True)
        df = self._complete_grid(df, cadence_minutes)
        had_obs = df["vehicle_count"].notna()
        n = len(df)

        # Optional calendar inputs default to "nothing special"
        for c in ("is_campus_event", "is_holiday", "event_load"):
            if c not in df.columns:
                df[c] = 0

        # 1. Cyclical Temporal Encodings
        hour = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
        dayofweek = df["timestamp"].dt.day_of_week

        df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
        df["day_sin"] = np.sin(2 * np.pi * dayofweek / 7.0)
        df["day_cos"] = np.cos(2 * np.pi * dayofweek / 7.0)
        df["is_weekend"] = (dayofweek >= 5).astype(int)

        # Zone identity (constant within a zone)
        zid = df["zone_id"].iloc[0] if "zone_id" in df.columns else None
        zc = zone_code(zid) if zid is not None else -1
        df["zone_code"] = zc

        # 2. Capacity-Relative Anchor
        df["occupancy_ratio"] = df["vehicle_count"] / df["operational_capacity"]

        # 3. Lags (shifted strictly into the past — per zone)
        for lag in self.lag_minutes:
            steps = lag // cadence_minutes
            df[f"lag_{lag}m"] = df["vehicle_count"].shift(steps)

        # 4. Inflow / Outflow Velocity
        step_15 = 15 // cadence_minutes
        step_30 = 30 // cadence_minutes
        df["delta_15m"] = df["vehicle_count"] - df["vehicle_count"].shift(step_15)
        df["delta_30m"] = df["vehicle_count"] - df["vehicle_count"].shift(step_30)

        # 5. Shifted Rolling Windows (excluding current step to avoid leakage)
        for w in self.rolling_windows_minutes:
            w_steps = w // cadence_minutes
            shifted = df["vehicle_count"].shift(1)
            df[f"rolling_mean_{w}m"] = shifted.rolling(window=w_steps, min_periods=1).mean()
            df[f"rolling_std_{w}m"] = shifted.rolling(window=w_steps, min_periods=1).std().fillna(0)

        # 6. Profile ("what is normal now") and deviation from it
        hol_now = df["is_holiday"].fillna(0).to_numpy()
        codes = np.full(n, zc)
        if self.profile is not None:
            df["profile_now"] = self.profile.lookup(codes, df["timestamp"], hol_now,
                                                    leave_week_out=self.profile_leave_week_out)
        else:
            df["profile_now"] = df["occupancy_ratio"]
        df["profile_dev"] = df["occupancy_ratio"] - df["profile_now"]

        # 7. Known-in-advance covariates at the arrival time t+h, and the direct targets
        weekday_now = (dayofweek < 5).to_numpy()
        for h in HORIZONS:
            steps = h // cadence_minutes
            fut_ts = df["timestamp"] + pd.Timedelta(minutes=h)
            fut_hol = df["is_holiday"].shift(-steps)
            fut_off = (fut_ts.dt.dayofweek >= 5).to_numpy() | (fut_hol.fillna(0).to_numpy() > 0)
            if self.profile is not None:
                df[horizon_col(h, "profile")] = self.profile.lookup(
                    codes, fut_ts, fut_hol.fillna(0).to_numpy(), leave_week_out=self.profile_leave_week_out)
            else:
                df[horizon_col(h, "profile")] = df["occupancy_ratio"]
            df[horizon_col(h, "event_load")] = df["event_load"].shift(-steps)
            df[horizon_col(h, "is_exam_day")] = df["is_exam_day"].shift(-steps)
            df[horizon_col(h, "is_holiday")] = fut_hol
            df[horizon_col(h, "mins_to_bell")] = mins_to_bell(fut_ts, fut_off)
            df[f"target_count_t{h}"] = df["vehicle_count"].shift(-steps)

        # Binary Saturation Target (>= saturation_threshold of capacity at +15m)
        df["target_saturated_t15"] = (
            df["target_count_t15"] >= (df["operational_capacity"] * self.saturation_threshold)
        ).astype(int)

        return df[had_obs.to_numpy()].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transform_zone_series(self, df: pd.DataFrame, cadence_minutes: int = 5) -> pd.DataFrame:
        """
        Transforms a (possibly multi-zone interleaved) DataFrame into ML-ready features.

        Required columns:
            timestamp, zone_id, vehicle_count, operational_capacity,
            is_exam_day, is_class_transition
        Optional: is_campus_event, is_holiday, event_load (default 0).
        """
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        if "zone_id" in df.columns and df["zone_id"].nunique() > 1:
            # Explicit loop (not groupby.apply): keeps the zone_id column on every
            # pandas version (pandas>=3 drops grouping columns inside apply).
            result = pd.concat(
                [self._transform_single_zone(g, cadence_minutes)
                 for _, g in df.groupby("zone_id", sort=True)],
                ignore_index=True,
            )
            result = result.reset_index(drop=True)
        else:
            result = self._transform_single_zone(df, cadence_minutes)

        return result

    def prepare_with_meta(
        self, df_features: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, pd.Series], pd.DataFrame]:
        """
        Like prepare_training_matrices but also returns a metadata frame
        (timestamp, zone_id, flags) aligned row-for-row with X. Rows are sorted
        chronologically (stable, so zones stay interleaved per timestamp), which
        is what time-series cross-validation needs.
        """
        target_cols = [f"target_count_t{h}" for h in HORIZONS]
        clean_df = (
            df_features.dropna(subset=FEATURE_COLS + target_cols)
            .sort_values("timestamp", kind="stable")
            .reset_index(drop=True)
        )
        X = clean_df[FEATURE_COLS]
        targets = {
            "H15": clean_df["target_count_t15"],
            "H30": clean_df["target_count_t30"],
            "H60": clean_df["target_count_t60"],
            "SAT15": clean_df["target_saturated_t15"],
        }
        meta_cols = ["timestamp", "zone_id"] + [c for c in ("sample_weight",) if c in clean_df.columns]
        meta = clean_df[meta_cols].copy()
        return X, targets, meta

    def prepare_training_matrices(
        self, df_features: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
        """
        Drops NaN rows created by lags and horizons, separates feature matrix X from targets.
        """
        target_cols = [f"target_count_t{h}" for h in HORIZONS]
        clean_df = (
            df_features.dropna(subset=FEATURE_COLS + target_cols)
            .copy().reset_index(drop=True)
        )

        X = clean_df[FEATURE_COLS]
        targets = {
            "H15": clean_df["target_count_t15"],
            "H30": clean_df["target_count_t30"],
            "H60": clean_df["target_count_t60"],
            "SAT15": clean_df["target_saturated_t15"],
        }

        return X, targets
