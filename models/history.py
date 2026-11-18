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

CADENCE_MIN = 5
RETENTION_MIN = 120
# An observation counts for a lag slot if within this tolerance of the slot time
TOLERANCE_S = 150


class ZoneHistory:
    """Thread-safe store of recent (timestamp, count) observations per zone."""

    def __init__(self):
        self._obs: Dict[str, Deque[Tuple[datetime, int]]] = defaultdict(deque)
        self._lock = Lock()

    def record(self, zone_id: str, ts: datetime, count: int) -> None:
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
        with self._lock:
            return self._at(zone_id, ts - timedelta(minutes=minutes))

    def window(self, zone_id: str, ts: datetime, minutes: int) -> List[int]:
        """Counts at t-5, t-10, ... t-minutes (the training rolling window)."""
        with self._lock:
            vals = []
            for k in range(1, minutes // CADENCE_MIN + 1):
                v = self._at(zone_id, ts - timedelta(minutes=k * CADENCE_MIN))
                if v is not None:
                    vals.append(v)
            return vals


# Process-wide buffer shared by the API and the CV counter
HISTORY = ZoneHistory()
