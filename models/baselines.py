import numpy as np
import pandas as pd
from typing import Dict, Tuple


class PersistenceForecaster:
    """
    Naive Baseline: y_hat(t+H) = y(t).
    Lower bound that quantifies the value-add of the ML models.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series):
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X["vehicle_count"].values


def _hour_day(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Recover integer hour-of-day / day-of-week from the sin/cos encodings."""
    hour = np.round((np.arctan2(df["hour_sin"], df["hour_cos"]) / (2 * np.pi) * 24) % 24).astype(int) % 24
    day = np.round((np.arctan2(df["day_sin"], df["day_cos"]) / (2 * np.pi) * 7) % 7).astype(int) % 7
    return np.asarray(day), np.asarray(hour)


class SeasonalHistoricalMeanForecaster:
    """
    Seasonal Baseline: y_hat(t+H) = mean(y | zone, day_of_week, hour).

    Zone-aware: each zone gets its own weekly profile, so a hostel lot is
    never averaged with a campus gate. Falls back to the zone mean, then the
    global mean, for unseen cells.
    """

    def __init__(self):
        self.lookup_table: Dict[Tuple[int, int, int], float] = {}
        self.zone_means: Dict[int, float] = {}
        self.global_mean = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series):
        day, hour = _hour_day(X)
        df = pd.DataFrame({
            "zone": X["zone_code"].values, "day": day, "hour": hour,
            "target": np.asarray(y),
        })
        self.lookup_table = df.groupby(["zone", "day", "hour"])["target"].mean().to_dict()
        self.zone_means = df.groupby("zone")["target"].mean().to_dict()
        self.global_mean = float(df["target"].mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        day, hour = _hour_day(X)
        zones = X["zone_code"].values
        return np.array([
            self.lookup_table.get(
                (int(z), int(d), int(h)),
                self.zone_means.get(int(z), self.global_mean),
            )
            for z, d, h in zip(zones, day, hour)
        ])
