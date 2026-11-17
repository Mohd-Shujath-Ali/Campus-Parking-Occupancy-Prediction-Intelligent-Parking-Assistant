"""
SHAP-based model interpretability for LPU Campus Parking XGBoost Forecaster.
Generates global beeswarm, feature importance, and local waterfall plots for the H15 horizon model.
"""
import os
import tempfile
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

import shap
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run_shap_analysis(
    model_path: str = "artifacts/pipelines/xgb_multi_horizon.joblib",
    test_data_path: str = "data/processed/lpu_timeseries.csv",
    output_dir: str = "artifacts/figures"
):
    os.makedirs(output_dir, exist_ok=True)

    print("[SHAP] Loading trained XGBoost multi-horizon pipeline...")
    xgb_model = joblib.load(model_path)
    h15_estimator = xgb_model.models["H15"]

    # Rebuild test split (must mirror train.py pipeline exactly)
    from data_pipeline.splitter import ChronologicalSplitter
    from features.feature_pipeline import TimeSeriesFeaturePipeline

    df_raw = pd.read_csv(test_data_path)
    splitter = ChronologicalSplitter(0.70, 0.15, 0.15, buffer_hours=1)
    _, _, df_test = splitter.split_dataframe(df_raw)
    feat_pipe = TimeSeriesFeaturePipeline()
    test_transformed = feat_pipe.transform_zone_series(df_test)
    X_test, targets_test = feat_pipe.prepare_training_matrices(test_transformed)

    # Subsample for explanation efficiency if test split is large
    if len(X_test) > 2000:
        np.random.seed(42)
        sample_indices = np.random.choice(len(X_test), size=2000, replace=False)
        X_sample = X_test.iloc[sample_indices].reset_index(drop=True)
    else:
        X_sample = X_test.reset_index(drop=True)

    print(f"[SHAP] Computing TreeSHAP values over {len(X_sample)} sample test records...")
    explainer = shap.TreeExplainer(h15_estimator)
    shap_vals_raw = explainer.shap_values(X_sample)

    # 1. Global Beeswarm Summary
    plt.figure(figsize=(10, 7), dpi=150)
    shap.summary_plot(shap_vals_raw, X_sample, show=False)
    plt.title("Global Feature Impact — XGBoost H15 (LPU Parking)", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_summary_beeswarm.png", bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir}/shap_summary_beeswarm.png")

    # 2. Local Waterfall — Rush Surge scenario (find row where delta_15m is highest)
    rush_idx = int(X_sample["delta_15m"].idxmax())
    base_val = explainer.expected_value
    if isinstance(base_val, np.ndarray):
        base_val = base_val[0]

    shap_exp_rush = shap.Explanation(
        values=shap_vals_raw[rush_idx],
        base_values=float(base_val),
        data=X_sample.iloc[rush_idx].values,
        feature_names=X_sample.columns.tolist()
    )
    plt.figure(figsize=(10, 6), dpi=150)
    shap.waterfall_plot(shap_exp_rush, show=False)
    plt.title("Local Explanation — Rush Surge Instance (Lecture Bell Transition)", fontsize=11, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_waterfall_rush_hour.png", bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir}/shap_waterfall_rush_hour.png")

    # 3. Local Waterfall — Steady State scenario (low delta, low occupancy at 2 PM)
    steady_mask = (X_sample["delta_15m"].abs() < 2) & (X_sample["occupancy_ratio"] < 0.4)
    steady_indices = steady_mask[steady_mask].index
    steady_idx = int(steady_indices[0]) if len(steady_indices) > 0 else 0

    shap_exp_steady = shap.Explanation(
        values=shap_vals_raw[steady_idx],
        base_values=float(base_val),
        data=X_sample.iloc[steady_idx].values,
        feature_names=X_sample.columns.tolist()
    )
    plt.figure(figsize=(10, 6), dpi=150)
    shap.waterfall_plot(shap_exp_steady, show=False)
    plt.title("Local Explanation — Steady State Instance (Low Activity Window)", fontsize=11, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_waterfall_steady_state.png", bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir}/shap_waterfall_steady_state.png")

    # 4. Bar plot: mean absolute SHAP values (feature importance)
    mean_shap = pd.Series(np.abs(shap_vals_raw).mean(axis=0), index=X_sample.columns)
    mean_shap = mean_shap.sort_values(ascending=True)
    plt.figure(figsize=(9, 7), dpi=150)
    ax = mean_shap.plot(kind="barh", color="#ea580c")
    plt.title("Mean |SHAP| Feature Importance — XGBoost H15", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Mean |SHAP| Value (Average Impact on Capacity Forecast)", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.5, axis="x")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_feature_importance_bar.png", bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir}/shap_feature_importance_bar.png")


if __name__ == "__main__":
    run_shap_analysis()
