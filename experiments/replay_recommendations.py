"""
Does following the recommendation actually find a parking slot?

Replays a held-out period. At every decision time (every `step` minutes, 07:00-19:00) a driver heading to one of
the campus blocks picks a ground using only what was known then, and we check what the lot really looked like
when they arrived (+15 / +30 / +60 minutes later):

  success = the chosen ground really had at least `free_slots_needed` free slots on arrival
  walk    = metres from the chosen ground to the destination
  cost    = walk + FAIL_COST_M x (1 - success): the driver's expected cost in walking-equivalent metres

Policies
  nearest               closest ground, ignoring occupancy (what most drivers do)
  nearest_open_now      closest ground that has space RIGHT NOW (looks at the lot, not the future)
  most_free_now         ground with most free slots right now, ignoring distance
  recommender           ParkingRanker using the model's forecast and calibrated free-slot chance
  oracle                closest ground that truly has space on arrival (unachievable upper bound)

Run:  python -m experiments.replay_recommendations              # report on the TEST period
      python -m experiments.replay_recommendations --tune       # choose ranker weights on the VALIDATION period
"""
import argparse
import itertools
import time
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd

from features.feature_pipeline import horizon_col
from models.data_prep import build_splits, split_raw
from models.trainer import load_artifacts, predict_clipped, regime_of
from models.uncertainty import event_flag, free_slots_needed
from recommendation.parking_ranker import ParkingRanker
from utils.config import load_zones_config

warnings.filterwarnings("ignore")
HORIZONS = {"H15": 15, "H30": 30, "H60": 60}
POLICIES = ["nearest", "nearest_open_now", "most_free_now", "recommender", "recommender_dispersed", "oracle"]
FAIL_COST_M = 500.0       # not finding a slot costs about as much as a 500 m detour (circling, re-parking, lateness)


class Context:
    """Everything needed to replay one split: forecasts for every row plus the real outcomes."""

    def __init__(self, split: str, csv: str = "data/processed/lpu_timeseries.csv", step_min: int = 15):
        self.models = load_artifacts()
        parts = split_raw(pd.read_csv(csv))
        splits = build_splits(parts[0], parts[1], parts[2], profile=self.models.profile)[0]
        sp = splits["val" if split == "val" else "test"]
        self.X, self.meta, self.y = sp.X.reset_index(drop=True), sp.meta.reset_index(drop=True), sp.y
        zl = load_zones_config()["zones"]
        self.zones = sorted(z["id"] for z in zl)
        self.zi = {z: i for i, z in enumerate(self.zones)}
        zmeta = {z["id"]: z for z in zl}
        self.dests = sorted({d for z in zmeta.values() for d in z["walking_distances_meters"]})
        self.dist = {d: np.array([zmeta[z]["walking_distances_meters"][d] for z in self.zones], float) for d in self.dests}
        X = self.X
        self.cap, self.cur = X.operational_capacity.to_numpy(float), X.vehicle_count.to_numpy(float)
        self.preds = predict_clipped(self.models.forecaster, X)
        self.pf, self.ps = {}, {}
        for k, h in HORIZONS.items():
            near = (X[horizon_col(h, "mins_to_bell")].to_numpy() <= 15).astype(float)
            ev = event_flag(X, h)
            self.pf[k] = self.models.resid.p_free(k, self.preds[k], self.cap, near, event=ev)
            self.ps[k] = self.models.resid.p_saturated(k, self.preds[k], self.cap, near, event=ev)
        self.regime = regime_of(X)
        ts = pd.to_datetime(self.meta.timestamp)
        sel = np.where(((ts.dt.minute % step_min == 0) & ts.dt.hour.between(7, 18)).to_numpy())[0]
        groups = pd.Series(sel).groupby(ts.iloc[sel].to_numpy()).apply(list)
        self.times = [(t, np.array(sorted(i, key=lambda j: self.zi[self.meta.zone_id.iloc[j]])))
                      for t, i in groups.items() if len(i) == len(self.zones)]


def simulate(ctx: Context, rankers: Dict[str, ParkingRanker], baselines: bool = True,
             horizons: List[str] = list(HORIZONS), anti_herding: bool = False) -> pd.DataFrame:
    rows = []
    for t, idx in ctx.times:
        cap, cur = ctx.cap[idx], ctx.cur[idx]
        need = np.array([free_slots_needed(c) for c in cap])
        reg = "event" if (ctx.X.event_load.to_numpy()[idx] > 0).any() else ctx.regime.iloc[idx[0]]
        for k in horizons:
            h = HORIZONS[k]
            actual = ctx.y[k].to_numpy(float)[idx]
            ok = actual <= cap - need
            good = np.where(ok)[0]
            open_now = np.where(cap - cur >= need)[0]
            pred_in = {ctx.zones[j]: {"predicted_count": int(round(ctx.preds[k][idx][j])),
                                      "saturation_prob": float(ctx.ps[k][idx][j]), "camera_age_seconds": 30.0,
                                      "p_free": float(ctx.pf[k][idx][j])} for j in range(len(ctx.zones))}
            for d in ctx.dests:
                dd = ctx.dist[d]
                chosen = {}
                if baselines:
                    chosen["nearest"] = int(np.argmin(dd))
                    chosen["most_free_now"] = int(np.argmax(cap - cur))
                    chosen["nearest_open_now"] = int(open_now[np.argmin(dd[open_now])]) if len(open_now) else chosen["nearest"]
                    chosen["oracle"] = int(good[np.argmin(dd[good])]) if len(good) else int(np.argmax(cap - actual))
                for name, rk in rankers.items():
                    top = rk.rank_zones(d, pred_in, horizon_minutes=h, apply_anti_herding=anti_herding)[0]["zone_id"]
                    chosen[name] = ctx.zi[top]
                for pol, j in chosen.items():
                    rows.append((pol, k, d, reg, bool(ok[j]), dd[j]))
    r = pd.DataFrame(rows, columns=["policy", "horizon", "dest", "regime", "success", "walk_m"])
    r["cost_m"] = r.walk_m + FAIL_COST_M * (~r.success)
    return r


def with_weights(**overrides) -> ParkingRanker:
    rk = ParkingRanker(seed=0)
    rk.weights = {**rk.weights, **overrides}
    return rk


def tune():
    t0 = time.time()
    ctx = Context("val", step_min=30)
    grid = {"availability_weight": [0.5, 0.35, 0.25], "distance_weight": [0.35, 0.5, 0.7], "p_free_blend": [0.6, 1.0]}
    names, rankers = [], {}
    for a, d, b in itertools.product(*grid.values()):
        n = f"a{a}_d{d}_b{b}"
        rankers[n] = with_weights(availability_weight=a, distance_weight=d, p_free_blend=b)
    r = simulate(ctx, rankers, baselines=True, horizons=["H30", "H60"])
    s = r.groupby("policy").agg(success_pct=("success", lambda x: 100 * x.mean()), walk_m=("walk_m", "mean"),
                                cost_m=("cost_m", "mean")).round(1)
    s["is_ranker"] = s.index.isin(list(rankers))
    pd.set_option("display.width", 200)
    print(f"VALIDATION period, {len(ctx.times)} decision times, {time.time() - t0:.0f}s (FAIL_COST_M={FAIL_COST_M:.0f})\n")
    print("Baselines:")
    print(s[~s.is_ranker].drop(columns="is_ranker").to_string())
    print("\nRanker weight grid (best 8 by expected cost):")
    print(s[s.is_ranker].drop(columns="is_ranker").sort_values("cost_m").head(8).to_string())
    print("\nCurrent configuration:")
    cur = with_weights()
    print(simulate(ctx, {"current": cur}, baselines=False, horizons=["H30", "H60"]).agg(
        success_pct=("success", lambda x: round(100 * x.mean(), 1)), walk_m=("walk_m", "mean"), cost_m=("cost_m", "mean")).T.round(1).to_string())
    s[s.is_ranker].sort_values("cost_m").to_csv("artifacts/recommender_tuning_val.csv")


def report():
    t0 = time.time()
    ctx = Context("test")
    r = simulate(ctx, {"recommender": ParkingRanker(seed=0)})
    # the same recommender with its anti-herding randomiser on (what drivers get in production): the price of
    # spreading demand across near-equal grounds. The replay cannot show the benefit, because simulated
    # drivers do not take slots from each other.
    r = pd.concat([r, simulate(ctx, {"recommender_dispersed": ParkingRanker(seed=0)}, baselines=False, anti_herding=True)])
    r["policy"] = pd.Categorical(r.policy, POLICIES, ordered=True)
    pd.set_option("display.width", 200)
    agg = lambda g: g.agg(success_pct=("success", lambda s: round(100 * s.mean(), 1)),
                          mean_walk_m=("walk_m", lambda s: round(s.mean())), expected_cost_m=("cost_m", lambda s: round(s.mean())))
    summary = agg(r.groupby(["horizon", "policy"], observed=True)).reset_index()
    print(f"TEST period: {len(ctx.times):,} decision times x {len(ctx.dests)} destinations x 3 arrival times, {time.time() - t0:.0f}s\n")
    for col, title in (("success_pct", "Drivers who found a free slot on arrival (%)"), ("mean_walk_m", "Average walk (m)"),
                       ("expected_cost_m", f"Expected cost (walk + {FAIL_COST_M:.0f} m x failure rate)")):
        print(title)
        print(summary.pivot(index="policy", columns="horizon", values=col).to_string(), "\n")
    by_reg = agg(r[r.horizon == "H30"].groupby(["regime", "policy"], observed=True)).reset_index()
    print("+30 min by kind of day: success % / average walk (m)")
    print(by_reg.pivot(index="regime", columns="policy", values="success_pct")[["nearest", "nearest_open_now", "recommender", "oracle"]].to_string())
    summary.to_csv("artifacts/recommendation_eval.csv", index=False)
    agg(r.groupby(["regime", "horizon", "policy"], observed=True)).reset_index().to_csv(
        "artifacts/recommendation_eval_by_regime.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true", help="grid-search ranker weights on the validation period")
    tune() if ap.parse_args().tune else report()
