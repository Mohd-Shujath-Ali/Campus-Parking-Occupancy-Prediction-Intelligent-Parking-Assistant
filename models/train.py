import os
import sys
import tempfile
if "MPLCONFIGDIR" not in os.environ:          # headless-safe matplotlib cache dir
    os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

import argparse
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import xgboost as xgb

from data_pipeline.generator import SCHEMA_COLUMNS, generate_lpu_parking_dataset
from models.baselines import PersistenceForecaster, ProfileForecaster, SeasonalHistoricalMeanForecaster
from models.data_prep import build_splits, split_raw
from models.demand import DemandForecaster, frame_from_raw
from models.forecaster import MultiHorizonForecaster
from models.pipeline import build_pipeline
from models.trainer import (KEYS, TrainedModels, build_meta, evaluate_core, fit_core, fit_demand, predict_clipped,
                            regime_table, save_artifacts)
from evaluation.metrics import compute_regression_metrics, compute_inflection_metrics
from evaluation.error_analysis import plot_residuals_vs_fitted, plot_error_distribution_qq, plot_error_by_zone
from utils.config import load_model_config


def _load_or_generate(data_csv: str, regen: bool) -> pd.DataFrame:
    if regen or not os.path.exists(data_csv):
        print("[Step 1] Generating the synthetic 18-week dataset (exams, events and holidays in every split)...")
        return generate_lpu_parking_dataset(output_csv_path=data_csv)
    df = pd.read_csv(data_csv)
    missing = [c for c in SCHEMA_COLUMNS if c not in df.columns]
    if missing:
        print(f"[Step 1] {data_csv} predates the calendar features (missing {missing}); regenerating...")
        return generate_lpu_parking_dataset(output_csv_path=data_csv)
    print("[Step 1] Loading existing time series dataset...")
    return df


def run_training_pipeline(grid_search: bool = False, fast: bool = False, cqr: bool = False,
                          regen: bool = False, data_csv: str = "data/processed/lpu_timeseries.csv"):
    print("=================================================================")
    print("   LPU CAMPUS PARKING INTELLIGENCE: END-TO-END TRAINING RUNNER   ")
    print("=================================================================")
    cfg = load_model_config()
    xgb_override = {"n_estimators": 150, "max_depth": 8} if fast else None
    rf_override = {"n_estimators": 40} if fast else None

    df_raw = _load_or_generate(data_csv, regen)
    print(f"Total raw observation records: {len(df_raw):,}  (SYNTHETIC: models learn the generator, not LPU)")

    print("\n[Step 2] Chronological split (70/15/15) BEFORE feature engineering...")
    df_train, df_val, df_test = split_raw(df_raw)
    print(f"Train {len(df_train):,} | Val {len(df_val):,} | Holdout test {len(df_test):,}")

    print("\n[Step 3] Normal-day profile (train only) and features (lags, calendar, arrival-time covariates)...")
    splits, profile = build_splits(df_train, df_val, df_test)
    tr, va, te = splits["train"], splits["val"], splits["test"]
    print(f"Feature matrix: {tr.X.shape[1]} features | profile built from {profile.n_rows:,} normal-day rows")

    if grid_search:
        print("\n[Step 3b] GridSearchCV on XGBoost over TimeSeriesSplit (5 folds, 1-hour purge)...")
        tscv = TimeSeriesSplit(n_splits=5, gap=12 * tr.X["zone_code"].nunique())
        grid = {"model__max_depth": [4, 6], "model__learning_rate": [0.03, 0.05],
                "model__subsample": [0.8, 1.0], "model__reg_lambda": [1.0, 2.0]}
        base = build_pipeline(xgb.XGBRegressor(n_estimators=100, objective="reg:squarederror", random_state=42))
        gs = GridSearchCV(base, grid, cv=tscv, scoring="neg_mean_absolute_error", n_jobs=-1, refit=False)
        gs.fit(tr.X, tr.y["H15"])      # rows are chronological (prepare_with_meta sorts by time)
        xgb_override = {k.replace("model__", ""): v for k, v in gs.best_params_.items()}
        print(f"GridSearch best params: {xgb_override} | CV MAE: {-gs.best_score_:.3f}")

    # ---- baselines and benchmark models ------------------------------------------------------
    print("\n[Step 4] Baselines: persistence, normal-day profile, seasonal mean...")
    pers = {k: PersistenceForecaster().predict(te.X) for k in KEYS}
    prof = {k: ProfileForecaster(k).predict(te.X) for k in KEYS}
    seas = SeasonalHistoricalMeanForecaster().fit(tr.X, tr.y["H15"])
    seas_pred = seas.predict(te.X)
    m = lambda k, p: compute_regression_metrics(te.y[k].values, p)
    for name, d in (("Persistence", pers), ("Normal-day profile", prof)):
        print(f"  {name:<20} " + "  ".join(f"{k} MAE={m(k, d[k])['MAE']}" for k in KEYS))

    print("\n[Step 4b] Benchmarks: Ridge (linear) and Random Forest (bagging)...")
    ridge = MultiHorizonForecaster("ridge"); ridge.fit(tr.X, tr.y); ridge_pred = ridge.predict(te.X)
    rf = MultiHorizonForecaster("random_forest", params=rf_override); rf.fit(tr.X, tr.y); rf_pred = rf.predict(te.X)
    for name, d in (("Ridge", ridge_pred), ("Random Forest", rf_pred)):
        print(f"  {name:<20} " + "  ".join(f"{k} MAE={m(k, d[k])['MAE']}" for k in KEYS))

    # ---- production models -------------------------------------------------------------------
    print("\n[Step 5] Production models: XGBoost x3 horizons, calibrated saturation classifier, residual distribution"
          + (", CQR intervals" if cqr else "") + "...")
    models, _ = fit_core(df_train, df_val, params=xgb_override, cqr=cqr, profile=profile)
    xgb_pred = predict_clipped(models.forecaster, te.X)
    val_pred = predict_clipped(models.forecaster, va.X)
    xgb_m = {k: m(k, xgb_pred[k]) for k in KEYS}
    xgb_val_h15 = compute_regression_metrics(va.y["H15"].values, val_pred["H15"])
    for k in KEYS:
        print(f"  XGBoost {k}: MAE={xgb_m[k]['MAE']} RMSE={xgb_m[k]['RMSE']} R2={xgb_m[k]['R2']}")
    print(f"  XGBoost val H15: MAE={xgb_val_h15['MAE']}  [validation sanity check]")

    print("\n[Step 6] Evaluation on the untouched holdout: skill, coverage and free-slot calibration...")
    ev = evaluate_core(models, te)
    for k, r in ev["horizons"].items():
        print(f"  {k}: MAE {r['MAE']} vs persistence {r['persistence_MAE']} ({r['skill_vs_persistence_pct']}% better) | "
              f"80% interval coverage {r['interval_coverage']} (events: {r['interval_coverage_events']}) width {r['interval_mean_width']} | "
              f"P(free slot) Brier {r['free_slot_brier']} vs climatology {r['free_slot_brier_climatology']}")
    reg = regime_table(models, te)
    print("\n  MAE by regime at +60 min (model vs persistence):")
    print(reg[reg.horizon == "H60"].drop(columns="horizon").to_string(index=False))
    os.makedirs("artifacts", exist_ok=True)
    reg.to_csv("artifacts/regime_metrics.csv", index=False)
    pd.concat([t.assign(horizon=k) for k, t in ev["calibration"].items()]).to_csv("artifacts/free_slot_calibration.csv", index=False)
    pd.DataFrame([{"Horizon": k, **{kk: v for kk, v in r.items() if kk.startswith("interval")}}
                  for k, r in ev["horizons"].items()]).to_csv("artifacts/interval_coverage.csv", index=False)

    print("\n[Step 6b] Inflection-hour error (lecture bell windows)...")
    rush = te.X["is_class_transition"].values
    infl = {n: compute_inflection_metrics(te.y["H15"].values, p, rush) for n, p in
            (("pers", pers["H15"]), ("ridge", ridge_pred["H15"]), ("rf", rf_pred["H15"]), ("xgb", xgb_pred["H15"]))}
    for n, v in infl.items():
        print(f"  {n:<6} rush MAE {v['MAE_Rush_Inflection']}  steady MAE {v['MAE_Steady_State']}")

    sat_eval = models.classifier.evaluate(te.y["SAT15"], models.classifier.predict(te.X),
                                          models.classifier.predict_proba(te.X))
    with open("artifacts/saturation_metrics.json", "w") as f:
        json.dump({**sat_eval, "threshold": 0.5, "calibrated": True}, f, indent=2)
    print(f"  Saturation classifier: Accuracy={sat_eval['Accuracy']} Recall={sat_eval['Recall']} PR_AUC={sat_eval['PR_AUC']}")

    # ---- demand forecaster (calendar only, days ahead) --------------------------------------------
    print("\n[Step 7] Demand forecaster (calendar features only)...")
    dem_eval = DemandForecaster().fit(df_train)
    y_ratio = (df_test["vehicle_count"] / df_test["operational_capacity"]).values
    Xc = frame_from_raw(df_test)
    p_dem = dem_eval.predict_ratio(Xc)
    seas_cal = SeasonalHistoricalMeanForecaster().fit(
        frame_from_raw(df_train), df_train["vehicle_count"] / df_train["operational_capacity"])
    p_seas = seas_cal.predict(Xc)
    t = df_test.assign(y=y_ratio, dem=p_dem, seas=p_seas, d=df_test["timestamp"].dt.date,
                       h=df_test["timestamp"].dt.hour)
    hits = {"dem": [], "seas": []}
    perr = {"dem": [], "seas": []}
    for _, g in t.groupby(["zone_id", "d"]):                     # is the predicted peak hour within +-1h of the real one?
        ah = g.loc[g["y"].idxmax(), "h"]
        for k in ("dem", "seas"):
            hits[k].append(abs(g.loc[g[k].idxmax(), "h"] - ah) <= 1)
            perr[k].append(abs(g[k].max() - g["y"].max()))
    dem_rows = []
    for name, k, pred in (("Calendar XGBoost", "dem", p_dem), ("Seasonal mean", "seas", p_seas)):
        dem_rows.append({"Model": name, "MAE_occupancy_pct": round(100 * float(np.abs(pred - y_ratio).mean()), 2),
                         "RMSE_occupancy_pct": round(100 * float(np.sqrt(((pred - y_ratio) ** 2).mean())), 2),
                         "peak_hour_within_1h": round(float(np.mean(hits[k])), 3),
                         "peak_level_MAE_pct": round(100 * float(np.mean(perr[k])), 2)})
        print("  ", dem_rows[-1])
    pd.DataFrame(dem_rows).to_csv("artifacts/demand_evaluation.csv", index=False)
    demand, demand_unc, demand_metrics = fit_demand(df_train, df_val, df_test)
    joblib.dump(demand_unc, "artifacts/pipelines/demand_uncertainty.joblib")

    # ---- artifacts ------------------------------------------------------------------------------
    print("\n[Step 8] Saving model artifacts and the model-version record...")
    os.makedirs("artifacts/pipelines", exist_ok=True)
    joblib.dump(ridge, "artifacts/pipelines/ridge_forecaster.joblib")
    joblib.dump(rf, "artifacts/pipelines/rf_forecaster.joblib", compress=3)
    joblib.dump(demand, "artifacts/pipelines/demand_forecaster.joblib")
    meta = build_meta(models, df_raw, "synthetic", {**ev["horizons"], "saturation_pr_auc": sat_eval["PR_AUC"],
                                                    "calendar_model": demand_metrics},
                      notes="Trained on the synthetic generator (data_pipeline/generator.py); retrain on real observations "
                            "with python -m models.retrain once cameras have collected enough data.")
    save_artifacts(models, meta)
    print(f"  version {meta['version']} | data_source={meta['data_source']}")

    # ---- diagnostics and benchmark table (read by /api/metrics and the analytics page) -----------
    print("\n[Step 9] Error analysis figures...")
    os.makedirs("artifacts/figures", exist_ok=True)
    plot_residuals_vs_fitted(te.y["H15"].values, xgb_pred["H15"], "Tuned XGBoost H15")
    plot_error_distribution_qq(te.y["H15"].values, xgb_pred["H15"], "Tuned XGBoost H15")
    test_frame = te.X.assign(zone_id=te.meta["zone_id"].values, **{"target_count_t15": te.y["H15"].values})
    plot_error_by_zone(test_frame, {"Tuned XGBoost": xgb_pred, "Ridge Regression": ridge_pred, "Random Forest": rf_pred}, te.y)

    print("\n[Step 10] Benchmark comparison table...")
    T, V = "Holdout Test", "Validation"
    rows = []
    def add(model, k, split, met, infl_mae=None):
        rows.append({"Model": model, "Horizon": f"+{KEYS[k]} min", "Split": split, "MAE": met["MAE"], "RMSE": met["RMSE"],
                     "R2": met["R2"], "Inflection_MAE": infl_mae,
                     "Skill_vs_Persistence_pct": round(100 * (1 - met["MAE"] / m(k, pers[k])["MAE"]), 1)})
    for k in KEYS:
        add("Persistence Baseline", k, T, m(k, pers[k]), infl["pers"]["MAE_Rush_Inflection"] if k == "H15" else None)
        add("Normal-day Profile", k, T, m(k, prof[k]))
    add("Seasonal Mean Baseline", "H15", T, m("H15", seas_pred))
    for k in KEYS:
        add("Ridge Regression", k, T, m(k, ridge_pred[k]), infl["ridge"]["MAE_Rush_Inflection"] if k == "H15" else None)
        add("Random Forest", k, T, m(k, rf_pred[k]), infl["rf"]["MAE_Rush_Inflection"] if k == "H15" else None)
    add("Tuned XGBoost", "H15", V, xgb_val_h15)
    for k in KEYS:
        add("Tuned XGBoost", k, T, xgb_m[k], infl["xgb"]["MAE_Rush_Inflection"] if k == "H15" else None)
    rows.append({"Model": "Saturation Classifier", "Horizon": "+15 min", "Split": T, "PR_AUC": sat_eval["PR_AUC"],
                 "Recall": sat_eval["Recall"], "Accuracy": sat_eval["Accuracy"]})
    comparison = pd.DataFrame(rows)
    comparison.to_csv("artifacts/model_comparison.csv", index=False)
    print(comparison.to_string(index=False))
    print("\nDone. Artifacts in artifacts/pipelines/, tables in artifacts/*.csv")
    return models, meta


def run_demand_only(data_csv: str = "data/processed/lpu_timeseries.csv"):
    """Refit only the calendar model and its uncertainty (e.g. after changing them) and record its metrics."""
    df_train, df_val, df_test = split_raw(pd.read_csv(data_csv))
    demand, unc, metrics = fit_demand(df_train, df_val, df_test)
    os.makedirs("artifacts/pipelines", exist_ok=True)
    joblib.dump(demand, "artifacts/pipelines/demand_forecaster.joblib")
    joblib.dump(unc, "artifacts/pipelines/demand_uncertainty.joblib")
    meta_path = Path("artifacts/pipelines/model_meta.json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        meta.setdefault("metrics_test", {})["calendar_model"] = metrics
        meta_path.write_text(json.dumps(meta, indent=2))
    print("Saved demand_forecaster.joblib and demand_uncertainty.joblib")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train all forecasting models and write artifacts")
    if "--demand-only" in sys.argv:
        run_demand_only()
        sys.exit(0)
    ap.add_argument("--grid-search", action="store_true", help="quick GridSearchCV before fitting")
    ap.add_argument("--fast", action="store_true", help="smaller models for development")
    ap.add_argument("--cqr", action="store_true", help="also fit feature-aware CQR intervals (slow, ~100 MB)")
    ap.add_argument("--regen", action="store_true", help="regenerate the synthetic dataset first")
    a = ap.parse_args()
    run_training_pipeline(grid_search=a.grid_search, fast=a.fast, cqr=a.cqr, regen=a.regen)
