"""
Calibrated uncertainty for the forecaster.

A point forecast of "112 of 120 slots full" is not what a driver needs; they need the CHANCE that
a slot is still free when they arrive. ResidualDistribution answers that empirically: it keeps the
distribution of (actual - forecast) measured on the held-out validation period, separately for each
horizon and for a few situations where the model is known to be more or less accurate (how full the
ground is, and whether a lecture bell is near). The chance of a free slot is then read straight off
that distribution, so it is calibrated on data the model was not trained on.
"""
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from features.feature_pipeline import HORIZONS, horizon_col

LEVELS = np.linspace(0.0, 1.0, 101)
MIN_BIN_SAMPLES = 400
MIN_EVENT_SAMPLES = 150                  # events are rare: a smaller bin is better than ignoring them
RATIO_EDGES = (0.5, 0.8, 0.95)           # predicted occupancy bins: <50%, 50-80%, 80-95%, >=95%
EVENT_OFFSET = 100                       # bin id offset for situations with a campus event
ALL_NORMAL, ALL_EVENT = -1, -2           # fallbacks: every non-event row / every event row


def free_slots_needed(capacity: float, min_abs: Optional[int] = None, min_frac: Optional[float] = None) -> int:
    """A ground counts as 'has space' when at least this many slots are free (rules in model_config.yaml)."""
    cfg = _unc_cfg()
    min_abs = cfg["free_slots_min_abs"] if min_abs is None else min_abs
    min_frac = cfg["free_slots_min_frac"] if min_frac is None else min_frac
    return int(max(min_abs, np.ceil(min_frac * capacity)))


def _unc_cfg() -> dict:
    from utils.config import load_model_config
    return load_model_config()["uncertainty"]


def clip_forecast(pred, current, capacity):
    """Counts are never negative and may overflow capacity by at most 30% (cars queuing / parking on verges)."""
    return np.clip(pred, 0.0, np.maximum(current, np.asarray(capacity) * 1.30))


def _bin_of(pred_ratio: np.ndarray, near_bell: np.ndarray, event: np.ndarray) -> np.ndarray:
    base = np.digitize(pred_ratio, RATIO_EDGES) * 2 + (np.asarray(near_bell) > 0).astype(int)
    return base + EVENT_OFFSET * (np.asarray(event) > 0).astype(int)


def event_flag(X: pd.DataFrame, horizon: int) -> np.ndarray:
    """An event is on now or at arrival time: the situation where forecasts are least reliable."""
    return ((X["event_load"].to_numpy() > 0) | (X[horizon_col(horizon, "event_load")].to_numpy() > 0)).astype(int)


class ResidualDistribution:
    def __init__(self):
        # horizon key -> {bin id -> 101 residual quantiles}. Fallbacks: ALL_NORMAL / ALL_EVENT.
        self.q: Dict[str, Dict[int, np.ndarray]] = {}
        self.n: Dict[str, Dict[int, int]] = {}

    def fit(self, X_val: pd.DataFrame, y_val: Dict[str, pd.Series], pred_val: Dict[str, np.ndarray]):
        cap = X_val["operational_capacity"].to_numpy(dtype=float)
        for key, h in (("H15", 15), ("H30", 30), ("H60", 60)):
            resid = np.asarray(y_val[key], dtype=float) - np.asarray(pred_val[key], dtype=float)
            ev = event_flag(X_val, h)
            bins = _bin_of(np.asarray(pred_val[key]) / cap, X_val[horizon_col(h, "mins_to_bell")].to_numpy() <= 15, ev)
            self.q[key] = {ALL_NORMAL: np.quantile(resid[ev == 0], LEVELS)}
            self.n[key] = {ALL_NORMAL: int((ev == 0).sum())}
            if (ev == 1).sum() >= MIN_EVENT_SAMPLES:
                self.q[key][ALL_EVENT] = np.quantile(resid[ev == 1], LEVELS)
                self.n[key][ALL_EVENT] = int((ev == 1).sum())
            for b in np.unique(bins):
                m = bins == b
                if m.sum() >= (MIN_EVENT_SAMPLES if b >= EVENT_OFFSET else MIN_BIN_SAMPLES):
                    self.q[key][int(b)] = np.quantile(resid[m], LEVELS)
                    self.n[key][int(b)] = int(m.sum())
        return self

    def _quantiles(self, key: str, pred: np.ndarray, cap: np.ndarray, near_bell: np.ndarray,
                   event: Optional[np.ndarray] = None) -> np.ndarray:
        """[n, 101] residual quantiles per row: its own situation if enough data, else all event / all normal rows."""
        ev = np.zeros(len(pred)) if event is None else np.atleast_1d(np.asarray(event))
        bins = _bin_of(pred / np.maximum(cap, 1), near_bell, ev)
        table = self.q[key]
        rows = []
        for b, e in zip(bins, ev):
            fallback = table.get(ALL_EVENT, table[ALL_NORMAL]) if e > 0 else table[ALL_NORMAL]
            rows.append(table.get(int(b), fallback))
        return np.stack(rows)

    def prob_at_most(self, key: str, pred, cap, near_bell, threshold, event=None) -> np.ndarray:
        """P(count <= threshold) for each row. All arguments are per-row arrays."""
        pred, cap, near_bell, threshold = (np.atleast_1d(np.asarray(a, dtype=float)) for a in
                                           (pred, cap, near_bell, threshold))
        qs = self._quantiles(key, pred, cap, near_bell, event)
        needed = threshold - pred                      # residual must be <= this
        out = np.empty(len(pred))
        for i in range(len(pred)):
            out[i] = np.interp(needed[i], qs[i], LEVELS, left=0.0, right=1.0)
        return out

    def p_free(self, key: str, pred, cap, near_bell, event=None) -> np.ndarray:
        """P(at least free_slots_needed slots are free at arrival)."""
        cap = np.atleast_1d(np.asarray(cap, dtype=float))
        need = np.array([free_slots_needed(c) for c in cap])
        return self.prob_at_most(key, pred, cap, near_bell, cap - need, event)

    def p_saturated(self, key: str, pred, cap, near_bell, level: float = 0.9, event=None) -> np.ndarray:
        cap = np.atleast_1d(np.asarray(cap, dtype=float))
        return 1.0 - self.prob_at_most(key, pred, cap, near_bell, level * cap - 1e-6, event)

    def interval(self, key: str, pred, cap, near_bell, coverage: float = 0.8, event=None) -> Tuple[np.ndarray, np.ndarray]:
        """Central `coverage` band around the forecast from the same distribution."""
        pred, cap, near_bell = (np.atleast_1d(np.asarray(a, dtype=float)) for a in (pred, cap, near_bell))
        qs = self._quantiles(key, pred, cap, near_bell, event)
        lo_i, hi_i = int(round(100 * (1 - coverage) / 2)), int(round(100 * (1 + coverage) / 2))
        return np.maximum(pred + qs[:, lo_i], 0.0), pred + qs[:, hi_i]


class CalendarResidualDistribution:
    """
    Uncertainty for the calendar-only demand model (used for arrival times hours ahead, where the live count no
    longer helps). Keeps the validation-period distribution of (actual - predicted) occupancy RATIO, by predicted
    level and whether an event is on, and turns it into vehicle ranges and free-slot chances for any ground.
    """

    def __init__(self):
        self.q: Dict[int, np.ndarray] = {}
        self.n: Dict[int, int] = {}

    def fit(self, pred_ratio, actual_ratio, event):
        pred_ratio, actual_ratio, event = (np.asarray(a, dtype=float) for a in (pred_ratio, actual_ratio, event))
        resid = actual_ratio - pred_ratio
        ev = (event > 0).astype(int)
        bins = np.digitize(pred_ratio, RATIO_EDGES) + EVENT_OFFSET * ev
        self.q = {ALL_NORMAL: np.quantile(resid[ev == 0], LEVELS)}
        self.n = {ALL_NORMAL: int((ev == 0).sum())}
        if (ev == 1).sum() >= MIN_EVENT_SAMPLES:
            self.q[ALL_EVENT] = np.quantile(resid[ev == 1], LEVELS)
            self.n[ALL_EVENT] = int((ev == 1).sum())
        for b in np.unique(bins):
            m = bins == b
            if m.sum() >= (MIN_EVENT_SAMPLES if b >= EVENT_OFFSET else MIN_BIN_SAMPLES):
                self.q[int(b)], self.n[int(b)] = np.quantile(resid[m], LEVELS), int(m.sum())
        return self

    def _quantiles(self, pred_ratio, event) -> np.ndarray:
        bins = np.digitize(pred_ratio, RATIO_EDGES) + EVENT_OFFSET * (np.asarray(event) > 0).astype(int)
        rows = []
        for b, e in zip(bins, np.asarray(event)):
            fallback = self.q.get(ALL_EVENT, self.q[ALL_NORMAL]) if e > 0 else self.q[ALL_NORMAL]
            rows.append(self.q.get(int(b), fallback))
        return np.stack(rows)

    def summarize(self, pred_ratio, cap, event, coverage: float = 0.8) -> Dict[str, np.ndarray]:
        """Vehicle-count range, P(free slot) and P(>=90% full) for each row."""
        pred_ratio, cap, event = (np.atleast_1d(np.asarray(a, dtype=float)) for a in (pred_ratio, cap, event))
        qs = self._quantiles(pred_ratio, event) * cap[:, None]            # residuals in vehicles
        point = pred_ratio * cap
        need = np.array([free_slots_needed(c) for c in cap])
        lo_i, hi_i = int(round(100 * (1 - coverage) / 2)), int(round(100 * (1 + coverage) / 2))
        p_free = np.array([np.interp(cap[i] - need[i] - point[i], qs[i], LEVELS, left=0.0, right=1.0) for i in range(len(point))])
        p_sat = np.array([1.0 - np.interp(0.9 * cap[i] - 1e-6 - point[i], qs[i], LEVELS, left=0.0, right=1.0) for i in range(len(point))])
        return {"pred": point, "low": np.maximum(point + qs[:, lo_i], 0.0), "high": point + qs[:, hi_i],
                "p_free": p_free, "p_sat": p_sat}


def reliability_table(p_pred: np.ndarray, outcome: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Predicted probability vs observed frequency, for calibration checks and the admin page."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p_pred, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if m.sum():
            rows.append({"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "n": int(m.sum()),
                         "mean_predicted": float(p_pred[m].mean()), "observed": float(outcome[m].mean())})
    return pd.DataFrame(rows)


def brier(p_pred: np.ndarray, outcome: np.ndarray) -> float:
    return float(np.mean((np.asarray(p_pred) - np.asarray(outcome)) ** 2))
