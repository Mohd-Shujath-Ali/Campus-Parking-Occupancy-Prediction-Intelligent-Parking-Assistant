"""
Parking Data Collection (core module 1).

Stores camera-derived vehicle counts in SQLite so that real observations can
replace the simulator. One row per (5-minute bucket, zone, camera); the zone
total is the SUM over the cameras that cover the zone (each camera counts only
its own ROI, cross-camera duplicates having been removed upstream by the
ground-plane NMS in computer_vision.occupancy_counter).
"""
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import pandas as pd

from utils.config import load_zones_config

# Serverless hosts (Vercel) have a read-only filesystem except /tmp, which is per-instance and ephemeral:
# collected counts there are a short-lived buffer, not durable storage. Set PARKING_DB to override.
_FALLBACK_DB = "/tmp/observations.db" if os.environ.get("VERCEL") else "data/live/observations.db"
DEFAULT_DB = os.environ.get("PARKING_DB", _FALLBACK_DB)
CADENCE_MIN = 5

SCHEMA = """
CREATE TABLE IF NOT EXISTS observations (
    bucket     TEXT NOT NULL,           -- 5-minute bucket start (naive local time, ISO)
    zone_id    TEXT NOT NULL,
    camera_id  TEXT NOT NULL,
    count      INTEGER NOT NULL CHECK (count >= 0),
    source     TEXT NOT NULL DEFAULT 'camera',
    received_at TEXT NOT NULL,
    PRIMARY KEY (bucket, zone_id, camera_id)
);
CREATE INDEX IF NOT EXISTS idx_obs_zone_bucket ON observations(zone_id, bucket);
"""


def floor_bucket(ts: datetime) -> datetime:
    """Floor to the 5-minute grid and drop tzinfo (the model works in campus-local time)."""
    ts = ts.replace(tzinfo=None) if ts.tzinfo else ts
    return ts.replace(minute=ts.minute - ts.minute % CADENCE_MIN, second=0, microsecond=0)


class ObservationStore:
    def __init__(self, path: str = DEFAULT_DB):
        self.path = path
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._mem = sqlite3.connect(":memory:", check_same_thread=False) if path == ":memory:" else None
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = self._mem or sqlite3.connect(self.path, timeout=10)
        try:
            yield conn
            conn.commit()
        finally:
            if self._mem is None:
                conn.close()

    def record(self, zone_id: str, camera_id: str, count: int, ts: datetime,
               source: str = "camera") -> Dict[str, object]:
        """Insert/replace one camera's count for its 5-minute bucket; return the zone total."""
        bucket = floor_bucket(ts)
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO observations VALUES (?,?,?,?,?,?)",
                (bucket.isoformat(), zone_id, camera_id, int(count), source,
                 datetime.now(timezone.utc).isoformat()),
            )
            total, n_cam = c.execute(
                "SELECT SUM(count), COUNT(*) FROM observations WHERE bucket=? AND zone_id=?",
                (bucket.isoformat(), zone_id)).fetchone()
        return {"bucket": bucket, "zone_total": int(total), "cameras_reporting": int(n_cam)}

    def zone_totals(self, since: Optional[datetime] = None, zone_id: Optional[str] = None) -> pd.DataFrame:
        """Zone-level counts per bucket: sum over the cameras of each zone."""
        q = ("SELECT bucket AS timestamp, zone_id, SUM(count) AS vehicle_count, "
             "COUNT(*) AS cameras FROM observations WHERE 1=1")
        args: List[object] = []
        if since is not None:
            q += " AND bucket >= ?"; args.append(since.isoformat())
        if zone_id:
            q += " AND zone_id = ?"; args.append(zone_id)
        q += " GROUP BY bucket, zone_id ORDER BY bucket, zone_id"
        with self._conn() as c:
            df = pd.read_sql_query(q, c, params=args)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    def stats(self) -> Dict[str, object]:
        with self._conn() as c:
            n, lo, hi = c.execute("SELECT COUNT(*), MIN(bucket), MAX(bucket) FROM observations").fetchone()
        return {"rows": n, "first_bucket": lo, "last_bucket": hi}


def export_training_frame(store: ObservationStore, out_csv: Optional[str] = None, min_cameras: int = 1,
                          entries=None) -> pd.DataFrame:
    """
    Convert collected counts to the schema of data/processed/lpu_timeseries.csv so they can be
    fed to the existing training pipeline unchanged. Missing 5-minute buckets are NOT
    interpolated (gaps stay gaps; the feature pipeline drops rows whose lags are missing).

    Exam / holiday / event / bell flags are not observable from cameras: they come from the campus calendar
    (`entries`, default: the registered calendar provider), the same rules the live forecaster uses.
    """
    from features.calendar import compute_flags
    caps = {z["id"]: z["operational_capacity"] for z in load_zones_config()["zones"]}
    df = store.zone_totals()
    df = df[df.cameras >= min_cameras].copy()
    df["operational_capacity"] = df.zone_id.map(caps)
    df = df.dropna(subset=["operational_capacity"]).reset_index(drop=True)
    df["operational_capacity"] = df["operational_capacity"].astype(int)
    if len(df):
        fl = compute_flags(df["timestamp"], df["zone_id"].tolist(), entries)
        for c in ("is_exam_day", "is_campus_event", "is_class_transition", "is_holiday", "event_load"):
            df[c] = fl[c].to_numpy()
    else:
        for c in ("is_exam_day", "is_campus_event", "is_class_transition", "is_holiday", "event_load"):
            df[c] = []
    df["camera_status"] = "live"
    df["camera_age_seconds"] = 0.0
    out = df[["timestamp", "zone_id", "vehicle_count", "operational_capacity", "is_exam_day", "is_campus_event",
              "is_class_transition", "is_holiday", "event_load", "camera_status", "camera_age_seconds"]]
    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_csv, index=False)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Export collected observations as a training CSV")
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--out", default="data/processed/lpu_live.csv")
    a = ap.parse_args()
    st = ObservationStore(a.db)
    print(st.stats())
    print(f"exported {len(export_training_frame(st, a.out))} zone-bucket rows -> {a.out}")
