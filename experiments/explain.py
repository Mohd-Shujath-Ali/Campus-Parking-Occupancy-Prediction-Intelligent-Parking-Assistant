"""
Explainability (rubric 3.8): global and local SHAP explanations of the deployed
+15 min XGBoost forecaster. Run after training:  python -m experiments.explain
"""
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from models.data_prep import load_splits
from models.pipeline import transformed_feature_names

warnings.filterwarnings("ignore")
FIG, OUT = "artifacts/figures", "artifacts/error_analysis"


def main():
    sp = load_splits(); te = sp["test"]
    pipe = joblib.load("artifacts/pipelines/xgb_multi_horizon.joblib").models["H15"]
    prep, model = pipe.named_steps["prep"], pipe.named_steps["model"]
    names = transformed_feature_names(pipe)

    idx = np.random.default_rng(0).choice(len(te.X), 4000, replace=False)
    Z = pd.DataFrame(prep.transform(te.X.iloc[idx]), columns=names)
    expl = shap.TreeExplainer(model)
    sv = expl(Z)

    # Merge the 12 zone one-hot columns into a single "zone identity" feature
    zone_cols = [i for i, n in enumerate(names) if n.startswith("zone=")]
    keep = [i for i in range(len(names)) if i not in zone_cols]
    vals = np.column_stack([sv.values[:, keep], sv.values[:, zone_cols].sum(1)])
    labels = [names[i] for i in keep] + ["zone identity"]
    imp = pd.Series(np.abs(vals).mean(0), index=labels).sort_values(ascending=False)
    imp.round(3).to_csv(f"{OUT}/shap_global_importance.csv", header=["mean_abs_shap_cars"])
    print("mean |SHAP| (cars):\n", imp.round(3).head(12).to_string())

    # Global: bar + beeswarm
    fig, ax = plt.subplots(figsize=(7, 5))
    imp.head(12)[::-1].plot.barh(ax=ax, color="#3b6fb6")
    ax.set(xlabel="mean |SHAP| (cars of prediction change)", title="Global feature importance (+15 min)")
    fig.tight_layout(); fig.savefig(f"{FIG}/shap_global_bar.png", dpi=150); plt.close(fig)

    plt.figure()
    shap.summary_plot(sv.values[:, keep], Z.iloc[:, keep], show=False, max_display=12)
    plt.tight_layout(); plt.savefig(f"{FIG}/shap_beeswarm.png", dpi=150, bbox_inches="tight"); plt.close()

    # Dependence plots for the features that matter most for dynamics
    for feat in ("delta_15m", "hour_sin"):
        plt.figure(figsize=(6, 4))
        shap.dependence_plot(feat, sv.values[:, keep], Z.iloc[:, keep], interaction_index=None, show=False)
        plt.tight_layout(); plt.savefig(f"{FIG}/shap_dependence_{feat}.png", dpi=150); plt.close()

    # Local: a bell-surge arrival vs a quiet weekend row
    meta = te.meta.reset_index(drop=True)
    X = te.X.reset_index(drop=True)
    pred = pipe.predict(X)
    cases = {
        "weekday_morning_surge": np.flatnonzero((X.is_class_transition == 1) & (X.is_weekend == 0)
                                                & (meta.timestamp.dt.hour == 8)
                                                & (meta.zone_id == "ZONE_MAINPARK_NORTH"))[10],
        "weekend_quiet": np.flatnonzero((X.is_weekend == 1) & (meta.timestamp.dt.hour == 11)
                                        & (meta.zone_id == "ZONE_MAINPARK_NORTH"))[10],
    }
    for label, i in cases.items():
        e = expl(pd.DataFrame(prep.transform(X.iloc[[i]]), columns=names))
        plt.figure()
        shap.plots.waterfall(e[0], max_display=9, show=False)
        plt.tight_layout(); plt.savefig(f"{FIG}/shap_local_{label}.png", dpi=150, bbox_inches="tight"); plt.close()
        r = X.iloc[i]
        print(f"\n[{label}] {meta.zone_id[i]} {meta.timestamp[i]}  current={r.vehicle_count:.0f} "
              f"lag15={r.lag_15m:.0f} delta15={r.delta_15m:.0f}  -> predicted {pred[i]:.1f} "
              f"(actual {te.y['H15'].reset_index(drop=True)[i]:.0f}); base={float(np.ravel(e.base_values)[0]):.1f}")
        top = pd.Series(e.values[0], index=names).sort_values(key=abs, ascending=False).head(6)
        print(top.round(2).to_string())


if __name__ == "__main__":
    main()
