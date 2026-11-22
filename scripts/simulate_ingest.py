"""
Feed camera-style counts into a RUNNING server, to try the live loop (real counts -> forecasts) without cameras.

    python -m scripts.simulate_ingest --user admin --password <admin password>
    python -m scripts.simulate_ingest --api-key <INGEST_API_KEY> --zones ZONE_BLOCK1 ZONE_LAW --once

Each tick posts one count per zone to POST /api/ingest, taken from the synthetic dataset at the matching weekday and
time (the same replay the demo feed uses), with a little noise. Run it for an hour or two, then open Model Health to
watch live forecasts being scored against these counts. Once any count has been ingested the server leaves demo mode.

These are SIMULATED counts: do not run this against a production server that should reflect real traffic.
"""
import argparse
import http.cookiejar
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime

import numpy as np

from models import simulator
from utils.config import load_zones_config


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--user")
    ap.add_argument("--password")
    ap.add_argument("--api-key", help="X-API-Key for /api/ingest (INGEST_API_KEY on the server); alternative to --user")
    ap.add_argument("--zones", nargs="*", help="default: all zones")
    ap.add_argument("--interval", type=int, default=300, help="seconds between ticks (default 300 = one 5-minute bucket)")
    ap.add_argument("--once", action="store_true", help="send one tick and exit")
    ap.add_argument("--noise", type=float, default=0.02, help="relative noise on each count")
    a = ap.parse_args()

    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))

    def post(path, body, headers=None):
        req = urllib.request.Request(a.url + path, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json", **(headers or {})})
        try:
            with opener.open(req, timeout=15) as r:
                return r.status, json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")

    headers = {}
    if a.api_key:
        headers["X-API-Key"] = a.api_key
    elif a.user and a.password:
        status, body = post("/api/auth/login", {"username": a.user, "password": a.password})
        if status != 200:
            sys.exit(f"Login failed: {body.get('detail')}")
    else:
        sys.exit("Give --user and --password, or --api-key.")

    zones = {z["id"]: z for z in load_zones_config()["zones"]}
    ids = a.zones or sorted(zones)
    rng = np.random.default_rng()
    while True:
        now = datetime.now().replace(second=0, microsecond=0)
        counts = simulator.counts_at([now], {z: zones[z] for z in ids})
        for z in ids:
            n = max(0, int(round(counts[z][0] * (1 + rng.normal(0, a.noise)))))
            status, _ = post("/api/ingest", {"zone_id": z, "count": n, "timestamp": now.isoformat(), "source": "simulator"}, headers)
            print(f"{now:%H:%M} {z:<22} {n:>4}  {'ok' if status == 200 else 'FAILED ' + str(status)}")
        if a.once:
            break
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
