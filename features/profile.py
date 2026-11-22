"""
Seasonal occupancy profile: what a zone normally looks like at this time of week.

profile(zone, day type, 15-minute slot) = mean occupancy RATIO over normal days (no exam, no event).
Day type is "weekend" on weekends and holidays, else "weekday".

This is the "past data by timing" the forecaster leans on. It is used three ways:
  * as features: the profile now, how far today deviates from it, and the profile at arrival time
  * as the base of the residual target (the model learns deviations from normal)
  * as the fallback forecast when a camera is stale or offline

Leakage control: rows the model is TRAINED on get a leave-one-week-out profile (their own week is
excluded), so the model cannot learn from a profile that already contains the value it predicts.
Validation, test and live rows use the full profile.
"""
from typing import Optional

import numpy as np
import pandas as pd

from features.calendar import zone_ids

SLOT_MIN = 15
N_SLOTS = 24 * 60 // SLOT_MIN
MIN_SAMPLES = 3            # fewer samples than this in a slot -> fall back to the day-type mean


def _slot(ts: pd.Series) -> np.ndarray:
    return ((ts.dt.hour * 60 + ts.dt.minute) // SLOT_MIN).to_numpy()


def _daytype(ts: pd.Series, is_holiday) -> np.ndarray:
    return ((ts.dt.dayofweek >= 5).to_numpy() | (np.asarray(is_holiday) > 0)).astype(int)


class ProfileModel:
    def __init__(self):
        self.n_zones = len(zone_ids())
        self.week0: Optional[pd.Timestamp] = None
        self.sum: Optional[np.ndarray] = None       # [zone, daytype, slot]
        self.cnt: Optional[np.ndarray] = None
        self.wsum: Optional[np.ndarray] = None      # [zone, daytype, slot, week]
        self.wcnt: Optional[np.ndarray] = None
        self.table: Optional[np.ndarray] = None     # mean ratio [zone, daytype, slot]
        self.n_rows = 0

    # ------------------------------------------------------------------ fit
    def fit(self, df: pd.DataFrame) -> "ProfileModel":
        ts = pd.to_datetime(df["timestamp"]).reset_index(drop=True)
        zc = np.array([zone_ids().index(z) for z in df["zone_id"]])
        ratio = (df["vehicle_count"] / df["operational_capacity"]).to_numpy(dtype=float)
        holiday = df["is_holiday"].to_numpy() if "is_holiday" in df else np.zeros(len(df))
        event = df["is_campus_event"].to_numpy() if "is_campus_event" in df else np.zeros(len(df))
        normal = (df["is_exam_day"].to_numpy() == 0) & (event == 0)
        if "event_load" in df:
            normal &= df["event_load"].to_numpy() == 0

        self.week0 = ts.min().normalize() - pd.Timedelta(days=int(ts.min().dayofweek))
        wk = ((ts - self.week0).dt.days // 7).to_numpy()
        n_weeks = int(wk.max()) + 1
        dt, sl = _daytype(ts, holiday), _slot(ts)
        weight = df["sample_weight"].to_numpy(dtype=float) if "sample_weight" in df else np.ones(len(df))
        z, d, s, w, r, sw = zc[normal], dt[normal], sl[normal], wk[normal], ratio[normal], weight[normal]

        shape = (self.n_zones, 2, N_SLOTS)
        self.wsum = np.zeros(shape + (n_weeks,))
        self.wcnt = np.zeros(shape + (n_weeks,))
        np.add.at(self.wsum, (z, d, s, w), r * sw)          # optional per-row weights: real counts can outweigh synthetic
        np.add.at(self.wcnt, (z, d, s, w), sw)
        self.sum, self.cnt = self.wsum.sum(axis=3), self.wcnt.sum(axis=3)
        self.table = self._means(self.sum, self.cnt)
        self.n_rows = int(normal.sum())
        return self

    @staticmethod
    def _means(total: np.ndarray, count: np.ndarray) -> np.ndarray:
        """Per-slot mean; thin slots fall back to the (zone, day type) mean."""
        zd_mean = total.sum(axis=2) / np.maximum(count.sum(axis=2), 1)            # [zone, daytype]
        out = np.where(count >= MIN_SAMPLES, total / np.maximum(count, 1), zd_mean[:, :, None])
        return out

    # ------------------------------------------------------------------ lookup
    def lookup(self, zone_codes, ts, is_holiday=None, leave_week_out: bool = False) -> np.ndarray:
        """Profile occupancy ratio for each (zone, timestamp)."""
        ts = pd.Series(pd.to_datetime(ts)).reset_index(drop=True)
        zc = np.asarray(zone_codes)
        zc_safe = np.where(zc >= 0, zc, 0)
        dt = _daytype(ts, np.zeros(len(ts)) if is_holiday is None else is_holiday)
        sl = _slot(ts)
        if not leave_week_out:
            return self.table[zc_safe, dt, sl]
        wk = ((ts - self.week0).dt.days // 7).to_numpy()
        inside = (wk >= 0) & (wk < self.wsum.shape[3])
        wk_safe = np.where(inside, wk, 0)
        tot = self.sum[zc_safe, dt, sl] - np.where(inside, self.wsum[zc_safe, dt, sl, wk_safe], 0.0)
        cnt = self.cnt[zc_safe, dt, sl] - np.where(inside, self.wcnt[zc_safe, dt, sl, wk_safe], 0.0)
        zd_tot = self.sum.sum(axis=2)[zc_safe, dt]
        zd_cnt = self.cnt.sum(axis=2)[zc_safe, dt]
        fallback = zd_tot / np.maximum(zd_cnt, 1)
        return np.where(cnt >= MIN_SAMPLES, tot / np.maximum(cnt, 1), fallback)

    def at(self, zone_id: str, ts, is_holiday: int = 0) -> float:
        """Scalar profile ratio (serving)."""
        ids = zone_ids()
        zc = ids.index(zone_id) if zone_id in ids else -1
        return float(self.lookup([zc], [ts], [is_holiday])[0])
