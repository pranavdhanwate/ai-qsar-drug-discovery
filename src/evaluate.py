from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculates standard regression performance metrics."""
    r2 = r2_score(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r_val, _ = pearsonr(y_true, y_pred) if len(y_true) > 2 else (0.0, 0.0)

    return {
        "r2": round(float(r2), 4),
        "rmse": round(float(rmse), 4),
        "mae": round(float(mae), 4),
        "pearson_r": round(float(r_val), 4),
    }


def compare_models(
    models_dict: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """Evaluates all trained models on train and independent test partitions."""
    rows = []
    for name, model in models_dict.items():
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)

        train_m = calculate_metrics(y_train, train_preds)
        test_m = calculate_metrics(y_test, test_preds)

        rows.append({
            "model": name,
            "train_r2": train_m["r2"],
            "train_rmse": train_m["rmse"],
            "test_r2": test_m["r2"],
            "test_rmse": test_m["rmse"],
            "test_mae": test_m["mae"],
            "test_pearson_r": test_m["pearson_r"],
        })

    return pd.DataFrame(rows)
