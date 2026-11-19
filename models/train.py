import os
import sys
import tempfile
if "MPLCONFIGDIR" not in os.environ:          # headless-safe matplotlib cache dir
    os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

import json
import yaml
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import xgboost as xgb

from data_pipeline.generator import generate_lpu_parking_dataset
from data_pipeline.splitter import ChronologicalSplitter
from features.feature_pipeline import TimeSeriesFeaturePipeline
from models.baselines import PersistenceForecaster, SeasonalHistoricalMeanForecaster
from models.forecaster import MultiHorizonForecaster
from models.classifier import SaturationClassifier
from evaluation.metrics import compute_regression_metrics, compute_inflection_metrics
from models.intervals import PredictionIntervals
from models.demand import DemandForecaster, frame_from_raw
from models.pipeline import build_pipeline
from evaluation.error_analysis import (
    plot_residuals_vs_fitted,
    plot_error_distribution_qq,
    plot_error_by_zone,
)
from utils.config import load_model_config, xgb_params


def run_training_pipeline(grid_search: bool = False):
    print("=================================================================")
    print("   LPU CAMPUS PARKING INTELLIGENCE: END-TO-END TRAINING RUNNER   ")
    print("=================================================================")

    cfg = load_model_config()
    data_cfg, feat_cfg = cfg["data"], cfg["features"]

    # 1. Dataset Generation / Loading
    data_csv = "data/processed/lpu_timeseries.csv"
    if not os.path.exists(data_csv):
        print("[Step 1] Synthesizing continuous 5-minute dataset for all 12 LPU zones...")
        df_raw = generate_lpu_parking_dataset()
    else:
        print("[Step 1] Loading existing continuous time series dataset...")
        df_raw = pd.read_csv(data_csv)

    print(f"Total Raw Observation Records: {len(df_raw)}")

    # 2. Chronological Split (70/15/15) — split BEFORE feature engineering to prevent leakage
    print("\n[Step 2] Executing Chronological Split with zero data leakage...")
    splitter = ChronologicalSplitter(
        train_ratio=data_cfg["train_ratio"],
        val_ratio=data_cfg["val_ratio"],
        test_ratio=data_cfg["test_ratio"],
        buffer_hours=data_cfg["chronological_buffer_hours"],
    )
    df_train, df_val, df_test = splitter.split_dataframe(df_raw)
    print(f"Train Records: {len(df_train)} | Val Records: {len(df_val)} | Holdout Test Records: {len(df_test)}")

    # 3. Feature Pipeline Transformation — groupby(zone_id) prevents cross-zone lag bleed
    print("\n[Step 3] Building Causal Lags, Velocity, Shifted Rolling Windows, and Cyclics...")
    feat_pipe = TimeSeriesFeaturePipeline(
        lag_minutes=feat_cfg["lag_minutes"],
        rolling_windows_minutes=feat_cfg["rolling_windows_minutes"],
        saturation_threshold=feat_cfg["saturation_threshold"],
    )

    # Each split transformed independently (no data flowing across the temporal gap)
    train_transformed = feat_pipe.transform_zone_series(df_train)
    val_transformed   = feat_pipe.transform_zone_series(df_val)
    test_transformed  = feat_pipe.transform_zone_series(df_test)

    X_train, targets_train = feat_pipe.prepare_training_matrices(train_transformed)
    X_val,   targets_val   = feat_pipe.prepare_training_matrices(val_transformed)
    X_test,  targets_test  = feat_pipe.prepare_training_matrices(test_transformed)

    print(f"Feature Matrix Shape: {X_train.shape} across {len(X_train.columns)} features.")

    # 4. Train Baselines
    print("\n[Step 4] Fitting Baselines (Persistence & Seasonal Historical Mean)...")
    persistence = PersistenceForecaster()
    persistence.fit(X_train, targets_train["H15"])
    y_pred_pers_h15 = persistence.predict(X_test)
    pers_metrics = compute_regression_metrics(targets_test["H15"].values, y_pred_pers_h15)
    print(f"Baseline Persistence (H15): MAE = {pers_metrics['MAE']}, RMSE = {pers_metrics['RMSE']}, R2 = {pers_metrics['R2']}")

    seasonal_mean = SeasonalHistoricalMeanForecaster()
    seasonal_mean.fit(X_train, targets_train["H15"])
    y_pred_seas_h15 = seasonal_mean.predict(X_test)
    seas_metrics = compute_regression_metrics(targets_test["H15"].values, y_pred_seas_h15)
    print(f"Baseline Seasonal Mean (H15): MAE = {seas_metrics['MAE']}, RMSE = {seas_metrics['RMSE']}, R2 = {seas_metrics['R2']}")

    # 4b. Linear baseline (Ridge) and 4c. bagging baseline (Random Forest); params from model_config.yaml
    print("\n[Step 4b] Training Linear Baseline (Ridge Regression)...")
    ridge_forecaster = MultiHorizonForecaster(model_type="ridge")
    ridge_forecaster.fit(X_train, targets_train)
    ridge_preds = ridge_forecaster.predict(X_test)
    ridge_metrics_h15 = compute_regression_metrics(targets_test["H15"].values, ridge_preds["H15"])
    print(f"Ridge Forecaster (H15): MAE = {ridge_metrics_h15['MAE']}, RMSE = {ridge_metrics_h15['RMSE']}, R2 = {ridge_metrics_h15['R2']}")

    print("\n[Step 4c] Training Non-Linear Ensemble Baseline (Random Forest)...")
    rf_forecaster = MultiHorizonForecaster(model_type="random_forest")
    rf_forecaster.fit(X_train, targets_train)
    rf_preds = rf_forecaster.predict(X_test)
    rf_metrics_h15 = compute_regression_metrics(targets_test["H15"].values, rf_preds["H15"])
    print(f"Random Forest Forecaster (H15): MAE = {rf_metrics_h15['MAE']}, RMSE = {rf_metrics_h15['RMSE']}, R2 = {rf_metrics_h15['R2']}")

    # 5. XGBoost. Default: hyperparameters from model_config.yaml, overridden by
    # configs/tuned_params.yaml (Optuna: python -m experiments.tune_xgboost).
    # Optional quick search: python -m models.train --grid-search (GridSearchCV over TimeSeriesSplit).
    xgb_override = None
    if grid_search:
        print("\n[Step 5] GridSearchCV on XGBoost over TimeSeriesSplit (5 folds, 1-hour purge)...")
        tscv = TimeSeriesSplit(n_splits=5, gap=12 * X_train["zone_code"].nunique())
        param_grid = {
            "model__max_depth": [4, 6],
            "model__learning_rate": [0.03, 0.05],
            "model__subsample": [0.8, 1.0],
            "model__reg_lambda": [1.0, 2.0],
        }
        base = build_pipeline(xgb.XGBRegressor(n_estimators=100, objective="reg:squarederror", random_state=42))
        # rows must be chronological for TimeSeriesSplit: order by the feature-pipeline timestamps
        order = np.argsort(train_transformed.dropna().reset_index(drop=True)["timestamp"].values, kind="stable")
        gs = GridSearchCV(base, param_grid, cv=tscv, scoring="neg_mean_absolute_error", n_jobs=-1, refit=False)
        gs.fit(X_train.iloc[order], targets_train["H15"].iloc[order])
        xgb_override = {k.replace("model__", ""): v for k, v in gs.best_params_.items()}
        print(f"GridSearch best params: {xgb_override} | CV MAE: {-gs.best_score_:.3f}")
    print("\n[Step 5b] Training Multi-Horizon Tuned XGBoost (H15, H30, H60)...")
    xgb_forecaster = MultiHorizonForecaster(model_type="xgboost", params=xgb_override)
    xgb_forecaster.fit(X_train, targets_train)
    preds     = xgb_forecaster.predict(X_test)
    val_preds = xgb_forecaster.predict(X_val)

    xgb_metrics_h15 = compute_regression_metrics(targets_test["H15"].values, preds["H15"])
    xgb_metrics_h30 = compute_regression_metrics(targets_test["H30"].values, preds["H30"])
    xgb_metrics_h60 = compute_regression_metrics(targets_test["H60"].values, preds["H60"])
    xgb_val_h15     = compute_regression_metrics(targets_val["H15"].values, val_preds["H15"])

    print(f"XGBoost Forecaster (H15): MAE = {xgb_metrics_h15['MAE']}, RMSE = {xgb_metrics_h15['RMSE']}, R2 = {xgb_metrics_h15['R2']}")
    print(f"XGBoost Forecaster (H30): MAE = {xgb_metrics_h30['MAE']}, RMSE = {xgb_metrics_h30['RMSE']}, R2 = {xgb_metrics_h30['R2']}")
    print(f"XGBoost Forecaster (H60): MAE = {xgb_metrics_h60['MAE']}, RMSE = {xgb_metrics_h60['RMSE']}, R2 = {xgb_metrics_h60['R2']}")
    print(f"XGBoost Val (H15):        MAE = {xgb_val_h15['MAE']}, R2 = {xgb_val_h15['R2']}  [validation sanity check]")

    # 6. Inflection-Hour Evaluation
    print("\n[Step 6] Computing Inflection-Hour MAE (Lecture Bell Surge Windows)...")
    pers_infl = compute_inflection_metrics(targets_test["H15"].values, y_pred_pers_h15, X_test["is_class_transition"].values)
    ridge_infl = compute_inflection_metrics(targets_test["H15"].values, ridge_preds["H15"], X_test["is_class_transition"].values)
    rf_infl = compute_inflection_metrics(targets_test["H15"].values, rf_preds["H15"], X_test["is_class_transition"].values)
    xgb_infl  = compute_inflection_metrics(targets_test["H15"].values, preds["H15"], X_test["is_class_transition"].values)

    print(f"Persistence Inflection MAE (Rush Hours): {pers_infl['MAE_Rush_Inflection']} cars")
    pers_r, xgb_r = pers_infl["MAE_Rush_Inflection"], xgb_infl["MAE_Rush_Inflection"]
    reduction = 100.0 * (pers_r - xgb_r) / pers_r if pers_r else 0.0
    print(f"Ridge Inflection MAE:         {ridge_infl['MAE_Rush_Inflection']} cars")
    print(f"Random Forest Inflection MAE: {rf_infl['MAE_Rush_Inflection']} cars")
    print(f"XGBoost Inflection MAE (Rush Hours):     {xgb_r} cars  ({reduction:.1f}% lower than persistence)")

    # 7. Train Dual-Head Saturation Classifier
    print("\n[Step 7] Training Dual-Head Saturation Classifier P(Occ >= 90%)...")
    sat_clf = SaturationClassifier()
    sat_clf.fit(X_train, targets_train["SAT15"], X_val, targets_val["SAT15"])
    sat_preds  = sat_clf.predict(X_test)
    sat_probas = sat_clf.predict_proba(X_test)
    sat_eval   = sat_clf.evaluate(targets_test["SAT15"], sat_preds, sat_probas)
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/saturation_metrics.json", "w") as f:
        json.dump({**sat_eval, "threshold": 0.5, "calibrated": True}, f, indent=2)
    print(f"Saturation Classifier: Accuracy={sat_eval['Accuracy']}, Recall={sat_eval['Recall']}, PR_AUC={sat_eval['PR_AUC']}")

    # 7b. Prediction intervals (conformalised quantile regression, calibrated on validation)
    print("\n[Step 7b] Fitting conformal prediction intervals (80%)...")
    intervals = PredictionIntervals(alpha=0.2).fit(X_train, targets_train, X_val, targets_val)
    cov_rows = []
    for h in intervals.horizons:
        cov = intervals.coverage(X_test, targets_test[h], h)
        cov_rows.append({"Horizon": h, "target_coverage": 0.80, **{k: round(v, 3) for k, v in cov.items()}})
        print(f"  {h}: test coverage = {cov['coverage']:.3f} (target 0.80), mean width = {cov['mean_width']:.1f} cars")
    pd.DataFrame(cov_rows).to_csv("artifacts/interval_coverage.csv", index=False)

    # 7c. Demand forecaster (calendar-only, days ahead). Evaluate fitted-on-train vs the
    # zone x weekday x hour seasonal mean, then refit on train+val for deployment so the
    # campus-event days (which fall in the validation period) are learned.
    print("\n[Step 7c] Demand forecaster (calendar features only)...")
    dem_eval = DemandForecaster().fit(df_train)
    y_ratio = (df_test["vehicle_count"] / df_test["operational_capacity"]).values
    Xc = frame_from_raw(df_test)
    p_dem = dem_eval.predict_ratio(Xc)
    seas = SeasonalHistoricalMeanForecaster().fit(
        frame_from_raw(df_train), df_train["vehicle_count"] / df_train["operational_capacity"])
    p_seas = seas.predict(Xc)
    # peak accuracy: per zone-day, is the predicted peak hour within +-1h of the actual one?
    t = df_test.assign(y=y_ratio, dem=p_dem, seas=p_seas, d=df_test["timestamp"].dt.date,
                       h=df_test["timestamp"].dt.hour)
    hits = {"dem": [], "seas": []}
    perr = {"dem": [], "seas": []}
    for _, g in t.groupby(["zone_id", "d"]):
        ah = g.loc[g.y.idxmax(), "h"]
        for k in ("dem", "seas"):
            hits[k].append(abs(g.loc[g[k].idxmax(), "h"] - ah) <= 1)
            perr[k].append(abs(g[k].max() - g.y.max()))
    dem_rows = []
    for name, k, pred in (("Calendar XGBoost", "dem", p_dem), ("Seasonal mean", "seas", p_seas)):
        dem_rows.append({"Model": name,
                         "MAE_occupancy_pct": round(100 * float(np.abs(pred - y_ratio).mean()), 2),
                         "RMSE_occupancy_pct": round(100 * float(np.sqrt(((pred - y_ratio) ** 2).mean())), 2),
                         "peak_hour_within_1h": round(float(np.mean(hits[k])), 3),
                         "peak_level_MAE_pct": round(100 * float(np.mean(perr[k])), 2)})
        print("  ", dem_rows[-1])
    pd.DataFrame(dem_rows).to_csv("artifacts/demand_evaluation.csv", index=False)
    demand = DemandForecaster().fit(pd.concat([df_train, df_val]))

    # 8. Export Model Artifacts
    # Persist wrappers via joblib (safe on the same runtime version).
    # Also save each inner XGBoost Booster in the portable .ubj format so the
    # models survive an XGBoost version upgrade or Docker deployment.
    os.makedirs("artifacts/pipelines", exist_ok=True)
    joblib.dump(ridge_forecaster, "artifacts/pipelines/ridge_forecaster.joblib")
    joblib.dump(rf_forecaster,    "artifacts/pipelines/rf_forecaster.joblib", compress=3)
    joblib.dump(xgb_forecaster, "artifacts/pipelines/xgb_multi_horizon.joblib")
    joblib.dump(sat_clf,        "artifacts/pipelines/saturation_classifier.joblib")
    joblib.dump(intervals,      "artifacts/pipelines/xgb_intervals.joblib")
    joblib.dump(demand,         "artifacts/pipelines/demand_forecaster.joblib")

    # Portable Booster exports (cross-version safe)
    booster_dir = Path("artifacts/pipelines/boosters")
    booster_dir.mkdir(parents=True, exist_ok=True)
    for horizon, model in xgb_forecaster.models.items():
        # models are Pipelines (preprocessing + XGBoost): export the XGBoost step
        model.named_steps["model"].get_booster().save_model(str(booster_dir / f"xgb_{horizon}.ubj"))
    sat_clf.base.named_steps["model"].get_booster().save_model(str(booster_dir / "saturation_clf.ubj"))
    print(f"\n[Step 8] Serialized joblib wrappers and portable .ubj boosters to artifacts/pipelines/")

    # 9. Error analysis & diagnostic figures
    print("\n[Step 9] Generating Error Analysis & Diagnostic Figures...")
    os.makedirs("artifacts/figures", exist_ok=True)
    plot_residuals_vs_fitted(targets_test["H15"].values, preds["H15"], "Tuned XGBoost H15")
    plot_error_distribution_qq(targets_test["H15"].values, preds["H15"], "Tuned XGBoost H15")
    plot_error_by_zone(
        test_transformed.dropna().reset_index(drop=True),
        {"Tuned XGBoost": preds, "Ridge Regression": ridge_preds, "Random Forest": rf_preds},
        targets_test,
    )

    # 10. Benchmark comparison table (read by /api/metrics and the analytics page)
    print("\n[Step 10] Building Benchmark Comparison Table...")
    T, V = "Holdout Test", "Validation"
    comparison_df = pd.DataFrame([
        {"Model": "Persistence Baseline",   "Horizon": "+15 min", "Split": T, "MAE": pers_metrics["MAE"], "RMSE": pers_metrics["RMSE"], "R2": pers_metrics["R2"], "Inflection_MAE": pers_infl["MAE_Rush_Inflection"]},
        {"Model": "Seasonal Mean Baseline", "Horizon": "+15 min", "Split": T, "MAE": seas_metrics["MAE"], "RMSE": seas_metrics["RMSE"], "R2": seas_metrics["R2"], "Inflection_MAE": None},
        {"Model": "Ridge Regression",       "Horizon": "+15 min", "Split": T, "MAE": ridge_metrics_h15["MAE"], "RMSE": ridge_metrics_h15["RMSE"], "R2": ridge_metrics_h15["R2"], "Inflection_MAE": ridge_infl["MAE_Rush_Inflection"]},
        {"Model": "Random Forest",          "Horizon": "+15 min", "Split": T, "MAE": rf_metrics_h15["MAE"], "RMSE": rf_metrics_h15["RMSE"], "R2": rf_metrics_h15["R2"], "Inflection_MAE": rf_infl["MAE_Rush_Inflection"]},
        {"Model": "Tuned XGBoost",          "Horizon": "+15 min", "Split": V, "MAE": xgb_val_h15["MAE"], "RMSE": xgb_val_h15["RMSE"], "R2": xgb_val_h15["R2"], "Inflection_MAE": None},
        {"Model": "Tuned XGBoost",          "Horizon": "+15 min", "Split": T, "MAE": xgb_metrics_h15["MAE"], "RMSE": xgb_metrics_h15["RMSE"], "R2": xgb_metrics_h15["R2"], "Inflection_MAE": xgb_infl["MAE_Rush_Inflection"]},
        {"Model": "Tuned XGBoost",          "Horizon": "+30 min", "Split": T, "MAE": xgb_metrics_h30["MAE"], "RMSE": xgb_metrics_h30["RMSE"], "R2": xgb_metrics_h30["R2"], "Inflection_MAE": None},
        {"Model": "Tuned XGBoost",          "Horizon": "+60 min", "Split": T, "MAE": xgb_metrics_h60["MAE"], "RMSE": xgb_metrics_h60["RMSE"], "R2": xgb_metrics_h60["R2"], "Inflection_MAE": None},
        {"Model": "Saturation Classifier",  "Horizon": "+15 min", "Split": T, "MAE": None, "RMSE": None, "R2": None, "Inflection_MAE": None,
         "PR_AUC": sat_eval["PR_AUC"], "Recall": sat_eval["Recall"], "Accuracy": sat_eval["Accuracy"]},
    ])
    comparison_df.to_csv("artifacts/model_comparison.csv", index=False)
    print("\nSaved artifacts/model_comparison.csv")
    print("\nRESULTS SUMMARY TABLE:")
    print(comparison_df.to_string(index=False))


if __name__ == "__main__":
    run_training_pipeline(grid_search="--grid-search" in sys.argv)
