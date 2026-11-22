"""
Error analysis (rubric 3.6) for the deployed +15 min forecaster and the
saturation classifier, on the untouched chronological test period.

Loads the SAME persisted pipelines that the API serves.
Run:  python -m experiments.error_analysis
"""
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)

from features.feature_pipeline import zone_order
from models.data_prep import load_splits

warnings.filterwarnings("ignore")
OUT, FIG = "artifacts/error_analysis", "artifacts/figures"
BUCKETS = [0, .25, .5, .75, .9, 10]
BUCKET_LABELS = ["<25%", "25-50%", "50-75%", "75-90%", ">=90%"]


def build_frame(te, model):
    pred = model.predict(te.X)
    df = te.meta.copy()
    zones = zone_order()
    df["cap"] = te.X["operational_capacity"].values
    df["current"] = te.X["vehicle_count"].values
    df["actual"] = te.y["H15"].values
    df["pred"] = pred
    df["persist"] = df["current"]
    df["err"] = df["pred"] - df["actual"]
    df["abs_err"] = df["err"].abs()
    df["abs_err_persist"] = (df["persist"] - df["actual"]).abs()
    df["hour"] = df["timestamp"].dt.hour
    df["weekend"] = te.X["is_weekend"].values
    df["transition"] = te.X["is_class_transition"].values
    df["exam"] = te.X["is_exam_day"].values
    df["event"] = te.X["is_campus_event"].values
    df["jump"] = (df["actual"] - df["current"]).abs()           # what persistence misses
    df["occ_bucket"] = pd.cut(df["actual"] / df["cap"], BUCKETS, labels=BUCKET_LABELS, right=False)
    return df


def grouped(df, by):
    g = df.groupby(by, observed=True).agg(
        n=("abs_err", "size"), MAE=("abs_err", "mean"), MAE_persistence=("abs_err_persist", "mean"),
        bias=("err", "mean"), cap=("cap", "mean"))
    g["MAE_pct_of_capacity"] = 100 * g["MAE"] / g["cap"]
    g["gain_vs_persistence_pct"] = 100 * (1 - g["MAE"] / g["MAE_persistence"])
    return g.drop(columns="cap").round(3)


def worst_cases(df, k=20):
    w = df.nlargest(k, "abs_err")[["timestamp", "zone_id", "current", "actual", "pred", "err",
                                   "transition", "jump", "cap"]].copy()
    def reason(r):
        tags = []
        if r.transition: tags.append("class-bell window")
        if r.jump >= 0.10 * r.cap: tags.append("large 15-min swing not visible in lags")
        if r.actual >= 0.9 * r.cap: tags.append("near full")
        return "; ".join(tags) or "noise spike"
    w["likely_reason"] = w.apply(reason, axis=1)
    return w.round(2)


def classifier_report(sp, clf):
    """Threshold is chosen on VALIDATION (max F1), then applied once to TEST."""
    pv = clf.predict_proba(sp["val"].X); yv = sp["val"].y["SAT15"].values
    pt = clf.predict_proba(sp["test"].X); yt = sp["test"].y["SAT15"].values
    prec, rec, thr = precision_recall_curve(yv, pv)
    f1s = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-9, None)
    t_star = float(thr[np.argmax(f1s)])
    rows = []
    for label, t in (("default 0.50", 0.5), (f"val-optimal {t_star:.2f}", t_star)):
        yp = (pt >= t).astype(int)
        rows.append({"threshold": label, "precision": precision_score(yt, yp),
                     "recall": recall_score(yt, yp), "F1": f1_score(yt, yp),
                     "PR_AUC": average_precision_score(yt, pt), "ROC_AUC": roc_auc_score(yt, pt),
                     "positive_rate_test": yt.mean()})
    rep = pd.DataFrame(rows).round(4)
    cm = confusion_matrix(yt, (pt >= t_star).astype(int))

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    p, r, _ = precision_recall_curve(yt, pt)
    ax[0].plot(r, p, lw=2); ax[0].set(xlabel="recall", ylabel="precision", title="Precision-recall (test)")
    ax[0].axhline(yt.mean(), ls="--", c="grey", label="no-skill"); ax[0].legend(); ax[0].grid(alpha=.3)
    fr, mp = calibration_curve(yt, pt, n_bins=10, strategy="quantile")
    ax[1].plot(mp, fr, "o-"); ax[1].plot([0, 1], [0, 1], "--", c="grey")
    ax[1].set(xlabel="mean predicted P(saturated)", ylabel="observed frequency",
              title="Calibration (test)"); ax[1].grid(alpha=.3)
    ax[2].imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax[2].text(j, i, int(v), ha="center", va="center", color="black")
    ax[2].set(xticks=[0, 1], yticks=[0, 1], xticklabels=["pred free", "pred sat."],
              yticklabels=["true free", "true sat."], title=f"Confusion @ {t_star:.2f}")
    fig.tight_layout(); fig.savefig(f"{FIG}/classifier_diagnostics.png", dpi=150); plt.close(fig)
    return rep, t_star, pt, yt


def figures(df):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    ax[0].hexbin(df.actual, df.pred, gridsize=45, bins="log", cmap="viridis")
    m = df.actual.max(); ax[0].plot([0, m], [0, m], "r--", lw=1)
    ax[0].set(xlabel="actual count (t+15)", ylabel="predicted", title="Predicted vs actual")
    ax[1].hist(df.err, bins=80, color="#3b6fb6"); ax[1].axvline(0, c="r", ls="--")
    ax[1].set(xlabel="error (pred - actual, cars)", title="Residuals")
    h = df.groupby("hour")[["abs_err", "abs_err_persist"]].mean()
    ax[2].plot(h.index, h.abs_err, lw=2, label="XGBoost"); ax[2].plot(h.index, h.abs_err_persist, lw=2, label="Persistence")
    ax[2].set(xlabel="hour of day", ylabel="MAE (cars)", title="Error by hour"); ax[2].legend(); ax[2].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{FIG}/error_overview.png", dpi=150); plt.close(fig)

    z = grouped(df, "zone_id").sort_values("MAE")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(z.index, z.MAE, label="XGBoost"); ax.barh(z.index, z.MAE_persistence, alpha=.35, label="Persistence")
    ax.set(xlabel="MAE (cars)", title="Error by zone"); ax.legend(); fig.tight_layout()
    fig.savefig(f"{FIG}/error_by_zone.png", dpi=150); plt.close(fig)

    # one zone, one weekday: actual vs forecast
    z0 = "ZONE_MAINPARK_NORTH"
    d = df[(df.zone_id == z0)]
    day = d[d.weekend == 0].timestamp.dt.date.iloc[len(d[d.weekend == 0]) // 2]
    d = d[d.timestamp.dt.date == day].sort_values("timestamp")
    fig, ax = plt.subplots(figsize=(10, 4))
    t = d.timestamp + pd.Timedelta(minutes=15)
    ax.plot(t, d.actual, "k", lw=2, label="actual"); ax.plot(t, d.pred, lw=2, label="XGBoost +15 min")
    ax.plot(t, d.persist, lw=1.2, ls="--", label="persistence")
    ax.set(title=f"{z0} on {day}", ylabel="vehicles"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{FIG}/forecast_example_day.png", dpi=150); plt.close(fig)


def main():
    sp = load_splits(); te = sp["test"]
    reg = joblib.load("artifacts/pipelines/xgb_multi_horizon.joblib").models["H15"]
    clf = joblib.load("artifacts/pipelines/saturation_classifier.joblib")
    df = build_frame(te, reg)
    tables = {
        "by_zone": grouped(df, "zone_id"), "by_hour": grouped(df, "hour"),
        "by_weekend": grouped(df, "weekend"), "by_class_transition": grouped(df, "transition"),
        "by_actual_occupancy": grouped(df, "occ_bucket"),
        "by_exam_day": grouped(df, "exam"), "by_campus_event": grouped(df, "event"),
    }
    for k, v in tables.items():
        v.to_csv(f"{OUT}/{k}.csv")
        print(f"\n--- {k} ---\n{v.to_string()}")
    w = worst_cases(df); w.to_csv(f"{OUT}/worst_20.csv", index=False)
    print("\n--- worst 20 ---\n", w.to_string(index=False))
    print("\nreason counts:\n", w.likely_reason.value_counts().to_string())
    q = df.abs_err.quantile([.5, .9, .95, .99]).round(2).to_dict()
    print(f"\nabs-error quantiles: {q}   overall MAE={df.abs_err.mean():.3f}  bias={df.err.mean():.3f}")
    figures(df)
    rep, t_star, _, _ = classifier_report(sp, clf)
    rep.to_csv(f"{OUT}/classifier_report.csv", index=False)
    print("\n--- saturation classifier (test) ---\n", rep.to_string(index=False))


if __name__ == "__main__":
    main()
