"""
Rolling per-zone occupancy history used at inference time.

Training features (lags, deltas, rolling mean/std) are computed from the past
60 minutes of 5-minute counts. At serving time the same quantities must come
from real past observations, not from the current count repeated, otherwise
delta=0 / std=0 would be fed to a model that rarely saw that pattern.
"""
from collections import defaultdict, deque
from datetime import datetime, timedelta
from threading import Lock
from typing import Deque, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

_IST = ZoneInfo("Asia/Kolkata")

CADENCE_MIN = 5
RETENTION_MIN = 120
# An observation counts for a lag slot if within this tolerance of the slot time
TOLERANCE_S = 150


def naive_ist(ts: datetime) -> datetime:
    """Campus-local (IST) wall-clock time without tzinfo. Naive input is assumed to be IST already."""
    return ts.astimezone(_IST).replace(tzinfo=None) if ts.tzinfo else ts


class ZoneHistory:
    """Thread-safe store of recent (timestamp, count) observations per zone."""

    def __init__(self):
        self._obs: Dict[str, Deque[Tuple[datetime, int]]] = defaultdict(deque)
        self._lock = Lock()

    def record(self, zone_id: str, ts: datetime, count: int) -> None:
        ts = naive_ist(ts)
        with self._lock:
            q = self._obs[zone_id]
            # Keep chronological order; ignore out-of-order older duplicates
            if q and ts <= q[-1][0]:
                return
            q.append((ts, int(count)))
            cutoff = ts - timedelta(minutes=RETENTION_MIN)
            while q and q[0][0] < cutoff:
                q.popleft()

    def clear(self) -> None:
        with self._lock:
            self._obs.clear()

    def _at(self, zone_id: str, target: datetime) -> Optional[int]:
        best, best_d = None, None
        for t, c in self._obs.get(zone_id, ()):
            d = abs((t - target).total_seconds())
            if d <= TOLERANCE_S and (best_d is None or d < best_d):
                best, best_d = c, d
        return best

    def lag(self, zone_id: str, ts: datetime, minutes: int) -> Optional[int]:
        ts = naive_ist(ts)
        with self._lock:
            return self._at(zone_id, ts - timedelta(minutes=minutes))

    def window(self, zone_id: str, ts: datetime, minutes: int) -> List[int]:
        """Counts at t-5, t-10, ... t-minutes (the training rolling window)."""
        ts = naive_ist(ts)
        with self._lock:
            vals = []
            for k in range(1, minutes // CADENCE_MIN + 1):
                v = self._at(zone_id, ts - timedelta(minutes=k * CADENCE_MIN))
                if v is not None:
                    vals.append(v)
            return vals


    def is_frozen(self, zone_id: str, readings: int = 6, min_span_min: int = 10) -> bool:
        """
        Detects a frozen camera/video feed: the last `readings` observations are identical,
        non-empty and span at least `min_span_min` minutes (so rapid polling cannot trigger it).
        """
        with self._lock:
            q = self._obs.get(zone_id)
            if not q or len(q) < readings:
                return False
            last = list(q)[-readings:]
            if (last[-1][0] - last[0][0]).total_seconds() < min_span_min * 60:
                return False
            counts = {c for _, c in last}
            return len(counts) == 1 and next(iter(counts)) > 0


# Process-wide buffer shared by the API and the CV counter
HISTORY = ZoneHistory()
