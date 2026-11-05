from pathlib import Path
from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_parity(
    y_train_true: np.ndarray,
    y_train_pred: np.ndarray,
    y_test_true: np.ndarray,
    y_test_pred: np.ndarray,
    output_path: str | Path,
    model_name: str = "XGBoost",
) -> Path:
    """Generates Parity Plot (Predicted vs Experimental pIC50) for Train and Test sets."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)

    # Diagonal unity line y = x
    all_vals = np.concatenate([y_train_true, y_test_true])
    v_min, v_max = float(np.min(all_vals)) - 0.5, float(np.max(all_vals)) + 0.5
    ax.plot([v_min, v_max], [v_min, v_max], "k--", alpha=0.7, label="Ideal Line (y = x)")

    # Scatter points
    ax.scatter(y_train_true, y_train_pred, color="#1976D2", alpha=0.6, s=50, label="Training Set", edgecolors="none")
    ax.scatter(y_test_true, y_test_pred, color="#D32F2F", alpha=0.85, s=65, label="Test Set", edgecolors="black", linewidth=0.7)

    # R2 and RMSE text
    from sklearn.metrics import r2_score, root_mean_squared_error
    test_r2 = r2_score(y_test_true, y_test_pred)
    test_rmse = root_mean_squared_error(y_test_true, y_test_pred)

    ax.text(
        0.05, 0.90,
        f"Test R² = {test_r2:.3f}\nTest RMSE = {test_rmse:.3f}",
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#ECEFF1", edgecolor="#90A4AE", alpha=0.9),
    )

    ax.set_xlabel("Experimental Bioactivity (pIC50)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Predicted Bioactivity (pIC50)", fontsize=11, fontweight="bold")
    ax.set_title(f"QSAR Parity Plot: {model_name} Model", fontsize=12, fontweight="bold")
    ax.set_xlim(v_min, v_max)
    ax.set_ylim(v_min, v_max)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path


def plot_williams(
    ad_df: pd.DataFrame,
    warning_leverage: float,
    output_path: str | Path,
) -> Path:
    """Generates Williams Plot for OECD Principle 3 Applicability Domain Evaluation."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)

    # Warning leverage line
    ax.axvline(warning_leverage, color="#D32F2F", linestyle="--", linewidth=1.5, label=f"Warning Leverage h* = {warning_leverage:.3f}")
    # Residual boundaries (+- 3 sigma)
    ax.axhline(3.0, color="#FF8F00", linestyle=":", linewidth=1.2, label="Residual Limits (±3σ)")
    ax.axhline(-3.0, color="#FF8F00", linestyle=":", linewidth=1.2)
    ax.axhline(0.0, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)

    # Points colored by domain status
    colors = ["#388E3C" if ok else "#C2185B" for ok in ad_df["within_ad"]]
    ax.scatter(ad_df["leverage"], ad_df["std_residual"], c=colors, s=60, edgecolors="black", linewidth=0.6, alpha=0.85)

    ax.set_xlabel("Leverage (h)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Standardized Residual (δ)", fontsize=11, fontweight="bold")
    ax.set_title("Williams Plot: Applicability Domain Evaluation", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    output_path: str | Path,
    top_n: int = 10,
) -> Path:
    """Plots top N most influential chemical descriptors in the QSAR model."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sorted_idx = np.argsort(importances)[::-1][:top_n]
    top_names = [feature_names[i] for i in sorted_idx]
    top_vals = [importances[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    y_pos = np.arange(len(top_names))
    ax.barh(y_pos, top_vals[::-1], color="#00695C", edgecolor="black", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Gini / Gain Importance Weight", fontsize=11, fontweight="bold")
    ax.set_title(f"Top {top_n} Molecular Features Driving Bioactivity", fontsize=12, fontweight="bold")
    ax.grid(axis="x", linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path
