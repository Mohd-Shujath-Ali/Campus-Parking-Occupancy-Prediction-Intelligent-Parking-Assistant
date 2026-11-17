import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any
from sklearn.base import BaseEstimator, TransformerMixin


class TimeSeriesFeaturePipeline:
    """
    Constructs causal time-series features (cyclics, lags, velocity, rolling windows)
    without future data leakage.

    FIX (bug #2): All shift()/rolling() operations are applied per-zone inside a
    groupby() so that lag values never bleed across zone boundaries when the
    DataFrame is multi-zone and interleaved (as produced by generator.py).
    """

    def __init__(
        self,
        lag_minutes: List[int] = [15, 30, 45, 60],
        rolling_windows_minutes: List[int] = [30, 60],
    ):
        self.lag_minutes = lag_minutes
        self.rolling_windows_minutes = rolling_windows_minutes

    # ------------------------------------------------------------------
    # Internal per-zone transform (called by groupby → no cross bleed)
    # ------------------------------------------------------------------

    def _transform_single_zone(self, df: pd.DataFrame, cadence_minutes: int) -> pd.DataFrame:
        """
        Applies all shift/rolling feature engineering on a single-zone
        DataFrame that is already sorted by timestamp.
        Called from transform_zone_series via groupby.apply.
        """
        df = df.copy().sort_values("timestamp").reset_index(drop=True)

        # 1. Cyclical Temporal Encodings
        hour = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
        dayofweek = df["timestamp"].dt.day_of_week

        df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
        df["day_sin"] = np.sin(2 * np.pi * dayofweek / 7.0)
        df["day_cos"] = np.cos(2 * np.pi * dayofweek / 7.0)
        df["is_weekend"] = (dayofweek >= 5).astype(int)

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

        # 6. Targets for Direct Multi-Horizon Forecasting
        df["target_count_t15"] = df["vehicle_count"].shift(-(15 // cadence_minutes))
        df["target_count_t30"] = df["vehicle_count"].shift(-(30 // cadence_minutes))
        df["target_count_t60"] = df["vehicle_count"].shift(-(60 // cadence_minutes))

        # Binary Saturation Target (>= 90% capacity at +15m)
        df["target_saturated_t15"] = (
            df["target_count_t15"] >= (df["operational_capacity"] * 0.90)
        ).astype(int)

        return df

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transform_zone_series(self, df: pd.DataFrame, cadence_minutes: int = 5) -> pd.DataFrame:
        """
        Transforms a (possibly multi-zone interleaved) DataFrame into ML-ready features.

        Required columns:
            timestamp, zone_id, vehicle_count, operational_capacity,
            is_exam_day, is_class_transition

        Multi-zone safety: groupby('zone_id') ensures shift()/rolling() never
        cross zone boundaries, eliminating the lag-bleed bug that corrupts
        training data when generator.py produces an interleaved DataFrame.
        """
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        if "zone_id" in df.columns and df["zone_id"].nunique() > 1:
            # Multi-zone path: apply per-zone, then concat
            chunks = []
            for zone_id, group in df.groupby("zone_id"):
                transformed_group = self._transform_single_zone(group, cadence_minutes)
                transformed_group["zone_id"] = zone_id
                chunks.append(transformed_group)
            result = pd.concat(chunks, ignore_index=True)
        else:
            # Single-zone path (also works if zone_id column is absent)
            result = self._transform_single_zone(df, cadence_minutes)

        return result

    def prepare_training_matrices(
        self, df_features: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
        """
        Drops NaN rows created by lags and horizons, separates feature matrix X from targets.
        """
        clean_df = df_features.dropna().copy().reset_index(drop=True)

        feature_cols = [
            "vehicle_count", "operational_capacity", "occupancy_ratio",
            "hour_sin", "hour_cos", "day_sin", "day_cos", "is_weekend",
            "is_exam_day", "is_class_transition",
            "lag_15m", "lag_30m", "lag_45m", "lag_60m",
            "delta_15m", "delta_30m",
            "rolling_mean_30m", "rolling_std_30m",
            "rolling_mean_60m", "rolling_std_60m",
        ]

        X = clean_df[feature_cols]
        targets = {
            "H15": clean_df["target_count_t15"],
            "H30": clean_df["target_count_t30"],
            "H60": clean_df["target_count_t60"],
            "SAT15": clean_df["target_saturated_t15"],
        }

        return X, targets
