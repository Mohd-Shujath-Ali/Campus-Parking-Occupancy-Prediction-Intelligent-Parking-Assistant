import os
import tempfile
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, List

def plot_residuals_vs_fitted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "XGBoost H15",
    output_path: str = "artifacts/figures/residuals_vs_fitted.png"
):
    """
    Plots residual values (y_true - y_pred) against fitted values (y_pred)
    to diagnose heteroscedasticity, non-linearity, and outlier influence.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    residuals = y_true - y_pred

    plt.figure(figsize=(9, 6), dpi=150)
    plt.scatter(y_pred, residuals, alpha=0.35, edgecolors='none', color='#ea580c', s=20)
    plt.axhline(0, color='#0f172a', linestyle='--', linewidth=1.5, label='Zero Error Reference')
    
    # Running median / trendline
    sorted_idx = np.argsort(y_pred)
    x_sort = y_pred[sorted_idx]
    res_sort = residuals[sorted_idx]
    if len(x_sort) > 100:
        window = len(x_sort) // 20
        rolling_mean = pd.Series(res_sort).rolling(window=window, center=True).mean()
        plt.plot(x_sort, rolling_mean, color='#dc2626', linewidth=2, label='Local Error Trend')

    plt.title(f"Residuals vs. Fitted Values — {model_name}", fontsize=12, fontweight='bold', pad=12)
    plt.xlabel("Fitted Values (Predicted Occupancy Count)", fontsize=10)
    plt.ylabel("Residuals (True − Predicted)", fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_error_distribution_qq(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "XGBoost H15",
    output_path: str = "artifacts/figures/error_distribution_qq.png"
):
    """
    Plots residual error histogram with normal overlay and Quantile-Quantile (Q-Q) plot
    to evaluate error normality and kurtosis (heavy tail behavior).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

    # 1. Error Distribution
    ax_hist = axes[0]
    count, bins, _ = ax_hist.hist(residuals, bins=40, density=True, alpha=0.6, color='#3b82f6', edgecolor='#1d4ed8')
    mu, std = stats.norm.fit(residuals)
    xmin, xmax = ax_hist.get_xlim()
    x = np.linspace(xmin, xmax, 200)
    p = stats.norm.pdf(x, mu, std)
    ax_hist.plot(x, p, 'r--', linewidth=2, label=f'Norm Fit (μ={mu:.2f}, σ={std:.2f})')
    ax_hist.set_title(f"Residual Error Distribution — {model_name}", fontsize=11, fontweight='bold')
    ax_hist.set_xlabel("Residual Error (Vehicles)", fontsize=10)
    ax_hist.set_ylabel("Density", fontsize=10)
    ax_hist.grid(True, linestyle=':', alpha=0.5)
    ax_hist.legend()

    # 2. Q-Q Plot
    ax_qq = axes[1]
    stats.probplot(residuals, dist="norm", plot=ax_qq)
    ax_qq.get_lines()[0].set_markerfacecolor('#0284c7')
    ax_qq.get_lines()[0].set_markeredgecolor('none')
    ax_qq.get_lines()[0].set_markersize(4)
    ax_qq.get_lines()[0].set_alpha(0.5)
    ax_qq.get_lines()[1].set_color('#dc2626')
    ax_qq.get_lines()[1].set_linewidth(2)
    ax_qq.set_title(f"Normal Q-Q Plot — {model_name}", fontsize=11, fontweight='bold')
    ax_qq.set_xlabel("Theoretical Quantiles", fontsize=10)
    ax_qq.set_ylabel("Ordered Residuals", fontsize=10)
    ax_qq.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_error_by_zone(
    test_transformed: pd.DataFrame,
    preds_dict: Dict[str, Dict[str, np.ndarray]],
    targets_test: Dict[str, pd.Series],
    output_path: str = "artifacts/figures/error_by_zone_boxplot.png"
):
    """
    Boxplot comparing absolute prediction errors across all 12 LPU campus zones
    to reveal spatial heteroscedasticity and highlight high-variance parking clusters.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if "zone_id" not in test_transformed.columns:
        print("plot_error_by_zone skipped: 'zone_id' column not present in transformed matrix.")
        return

    zones = test_transformed["zone_id"].values
    y_true = targets_test["H15"].values

    records = []
    for model_name, preds in preds_dict.items():
        if "H15" in preds:
            abs_err = np.abs(y_true - preds["H15"])
            for z, err in zip(zones, abs_err):
                records.append({"Zone": z, "Absolute_Error": err, "Model": model_name})

    if not records:
        return

    df_err = pd.DataFrame(records)
    unique_zones = sorted(df_err["Zone"].unique())
    models = sorted(df_err["Model"].unique())

    plt.figure(figsize=(14, 6), dpi=150)
    
    # We can plot grouped boxplots or average error bars per zone
    mean_errors = df_err.groupby(["Zone", "Model"])["Absolute_Error"].mean().unstack()
    
    ax = mean_errors.plot(kind="bar", figsize=(14, 6), width=0.8, colormap="viridis")
    plt.title("Spatial Error Breakdown: Mean Absolute Error (MAE) Across Campus Zones (H15)", fontsize=12, fontweight='bold', pad=12)
    plt.xlabel("LPU Campus Parking Zone", fontsize=10)
    plt.ylabel("Mean Absolute Error (Vehicles)", fontsize=10)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.grid(True, linestyle=':', alpha=0.5, axis='y')
    plt.legend(title="Algorithm", frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
