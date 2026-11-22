"""
How should a forecast behave when the driver picks an arrival time hours ahead (e.g. 5 PM, asked at 9 AM)?

The live models stop at +60 min. Beyond that, compare (MAE in vehicles at the arrival time):

  persistence         the count right now
  profile_decay_old   what the code did before: decay the +60 deviation toward the normal-day profile at t+60
  profile_at_arrival  normal-day profile at the arrival time (ignores exams / events / holidays)
  calendar            calendar-only demand model at the arrival time (knows exams, holidays, events)
  blend(tau)          calendar(arrival) + (live +60 forecast - calendar(t+60)) x exp(-(h-60)/tau)

tau is chosen on the VALIDATION period and reported on the TEST period.
Run:  python -m experiments.far_horizon
"""
import warnings

import numpy as np
import pandas as pd

from features.feature_pipeline import zone_code
from models.data_prep import build_splits, split_raw
from models.demand import DemandForecaster, frame_from_raw
from models.trainer import load_artifacts, predict_clipped, regime_of

warnings.filterwarnings("ignore")
HOURS = [1.5, 2, 3, 4, 6, 8, 12]
TAUS = [15, 30, 45, 60, 90, 120, 180, 360]


def table(split, raw, models, dem):
    """Per (anchor row, horizon): actual count at arrival and each method's forecast."""
    X, meta = split.X.reset_index(drop=True), split.meta.reset_index(drop=True)
    raw = raw.copy()
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    raw["cal"] = dem.predict_ratio(frame_from_raw(raw)) * raw["operational_capacity"]
    raw["prof"] = models.profile.lookup(raw.zone_id.map(zone_code).to_numpy(), raw.timestamp,
                                        raw["is_holiday"].to_numpy()) * raw["operational_capacity"]
    idx = raw.set_index(["zone_id", "timestamp"])[["vehicle_count", "cal", "prof", "event_load", "is_exam_day", "is_holiday"]]
    live60 = predict_clipped(models.forecaster, X)["H60"]
    keep = (pd.to_datetime(meta.timestamp).dt.minute % 30 == 0).to_numpy()
    anchors = pd.DataFrame({"zone_id": meta.zone_id[keep].values, "t": pd.to_datetime(meta.timestamp[keep]).values,
                            "now": X.vehicle_count[keep].values, "live60": live60[keep]})
    at60 = idx.reindex(list(zip(anchors.zone_id, anchors.t + pd.Timedelta(minutes=60))))
    anchors["cal60"], anchors["prof60"] = at60["cal"].values, at60["prof"].values
    out = []
    for hrs in HOURS:
        m = int(hrs * 60)
        arr = idx.reindex(list(zip(anchors.zone_id, anchors.t + pd.Timedelta(minutes=m))))
        d = anchors.assign(h=m, actual=arr["vehicle_count"].values, cal_arr=arr["cal"].values, prof_arr=arr["prof"].values,
                           special=((arr["event_load"] > 0) | (arr["is_exam_day"] > 0) | (arr["is_holiday"] > 0)).values)
        out.append(d.dropna(subset=["actual", "cal60", "cal_arr"]))
    return pd.concat(out, ignore_index=True)


def methods(d, tau=None):
    w = lambda t: np.exp(-(d.h - 60) / t)
    res = {"persistence": d.now,
           "profile_decay_old": d.prof60 + (d.live60 - d.prof60) * np.exp(-(d.h - 60) / 45.0),
           "profile_at_arrival": d.prof_arr,
           "calendar": d.cal_arr}
    for t in ([tau] if tau else TAUS):
        res[f"blend_tau{t}"] = d.cal_arr + (d.live60 - d.cal60) * w(t)
    return res


def main():
    models = load_artifacts()
    df = pd.read_csv("data/processed/lpu_timeseries.csv")
    tr, va, te = split_raw(df)
    dem = DemandForecaster().fit(tr)                      # train only: val/test stay out of sample
    sp = build_splits(tr, va, te, profile=models.profile)[0]

    dv = table(sp["val"], va, models, dem)
    mae_v = {k: float(np.abs(dv.actual - v).mean()) for k, v in methods(dv).items()}
    best = min((k for k in mae_v if k.startswith("blend")), key=mae_v.get)
    tau = int(best.replace("blend_tau", ""))
    print("VALIDATION MAE (all horizons pooled):", {k: round(v, 2) for k, v in sorted(mae_v.items(), key=lambda x: x[1])})
    print(f"-> chosen tau = {tau} min\n")

    dt = table(sp["test"], te, models, dem)
    rows = []
    for h, g in dt.groupby("h"):
        for k, v in methods(g, tau).items():
            rows.append({"hours_ahead": h / 60, "method": k, "MAE": round(float(np.abs(g.actual - v).mean()), 2),
                         "MAE_special_days": round(float(np.abs(g.actual - v)[g.special].mean()), 2) if g.special.any() else None})
    r = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("TEST MAE (vehicles) by hours ahead")
    print(r.pivot(index="method", columns="hours_ahead", values="MAE").to_string())
    print("\nTEST MAE on exam / holiday / event arrival times")
    print(r.pivot(index="method", columns="hours_ahead", values="MAE_special_days").to_string())
    r.to_csv("artifacts/far_horizon.csv", index=False)
    with open("artifacts/far_horizon_tau.txt", "w") as f:
        f.write(str(tau))


if __name__ == "__main__":
    main()
