"""
Are feature-aware CQR intervals worth their cost over the free residual-distribution intervals?

Both are calibrated on the validation period and judged on the untouched test period at the same nominal
80% coverage; a better method gives the same coverage with a narrower band.

Run:  python -m experiments.interval_comparison
"""
import numpy as np
import pandas as pd

from models.data_prep import split_raw
from models.trainer import evaluate_core, fit_core, regime_of
from data_pipeline.generator import generate_lpu_parking_dataset
import os


def main():
    csv = "data/processed/lpu_timeseries.csv"
    df = pd.read_csv(csv) if os.path.exists(csv) else generate_lpu_parking_dataset()
    tr, va, te = split_raw(df)
    models, splits = fit_core(tr, va, params={"n_estimators": 150, "max_depth": 8}, cqr=True)
    from models.data_prep import build_splits
    test = build_splits(tr, va, te, profile=models.profile)[0]["test"]

    with_cqr = evaluate_core(models, test)
    cqr, models.intervals = models.intervals, None
    resid = evaluate_core(models, test)
    rows = []
    for k in ("H15", "H30", "H60"):
        for name, r in (("residual distribution", resid), ("CQR", with_cqr)):
            h = r["horizons"][k]
            rows.append({"horizon": k, "method": name, "coverage": h["interval_coverage"], "mean_width": h["interval_mean_width"]})
    out = pd.DataFrame(rows)
    print(out.pivot(index="horizon", columns="method", values=["coverage", "mean_width"]).to_string())

    # coverage by regime, where a feature-aware band should help (events, exams)
    from models.trainer import predict_clipped
    from features.feature_pipeline import horizon_col
    X = test.X
    reg = regime_of(X)
    cap = X["operational_capacity"].to_numpy(float)
    p = predict_clipped(models.forecaster, X)["H60"]
    y = test.y["H60"].to_numpy(float)
    near = (X[horizon_col(60, "mins_to_bell")].to_numpy() <= 15).astype(float)
    from models.uncertainty import event_flag
    lo_r, hi_r = models.resid.interval("H60", p, cap, near, 0.8, event=event_flag(X, 60))
    lo_c, hi_c = cqr.predict(X, "H60", point=p)
    rr = []
    for r in reg.unique():
        m = (reg == r).to_numpy()
        rr.append({"regime": r, "rows": int(m.sum()),
                   "resid_cov": round(float(((y[m] >= lo_r[m]) & (y[m] <= hi_r[m])).mean()), 3),
                   "resid_width": round(float((hi_r[m] - lo_r[m]).mean()), 1),
                   "cqr_cov": round(float(((y[m] >= lo_c[m]) & (y[m] <= hi_c[m])).mean()), 3),
                   "cqr_width": round(float((hi_c[m] - lo_c[m]).mean()), 1)})
    print("\n+60 min, by regime")
    print(pd.DataFrame(rr).to_string(index=False))
    out.to_csv("artifacts/interval_comparison.csv", index=False)


if __name__ == "__main__":
    main()
