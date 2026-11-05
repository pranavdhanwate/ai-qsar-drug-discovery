from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd


class ApplicabilityDomain:
    """
    Evaluates OECD Principle 3 Applicability Domain using Leverage (Hat Matrix)
    and standardized residuals (Williams Plot methodology).
    """

    def __init__(self, warning_multiplier: float = 3.0, residual_limit: float = 3.0):
        self.warning_multiplier = warning_multiplier
        self.residual_limit = residual_limit
        self.hat_matrix_inv_: np.ndarray | None = None
        self.warning_leverage_: float = 0.0
        self.residual_std_: float = 1.0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, y_train_pred: np.ndarray) -> None:
        n, p = X_train.shape
        # Use pseudo-inverse for numerical stability with multicollinear features
        X_pinv = np.linalg.pinv(X_train.T @ X_train)
        self.hat_matrix_inv_ = X_pinv
        # Warning leverage: h* = 3(p + 1) / n
        self.warning_leverage_ = (self.warning_multiplier * (p + 1)) / n

        residuals = y_train - y_train_pred
        self.residual_std_ = float(np.std(residuals)) if np.std(residuals) > 0 else 1.0

    def compute_leverage(self, X: np.ndarray) -> np.ndarray:
        """Calculates leverage (diagonal elements of hat matrix) for any set of compounds."""
        if self.hat_matrix_inv_ is None:
            raise RuntimeError("Applicability domain not fitted.")

        leverages = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            xi = X[i, :]
            h_val = xi.T @ self.hat_matrix_inv_ @ xi
            leverages[i] = max(0.0, float(h_val))
        return leverages

    def evaluate_test_set(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        y_test_pred: np.ndarray,
    ) -> pd.DataFrame:
        """Determines leverage and standardized residuals for test set molecules."""
        leverages = self.compute_leverage(X_test)
        raw_residuals = y_test - y_test_pred
        std_residuals = raw_residuals / self.residual_std_

        within_ad = (leverages <= self.warning_leverage_) & (np.abs(std_residuals) <= self.residual_limit)

        ad_status = []
        for h, res in zip(leverages, std_residuals):
            if h <= self.warning_leverage_ and abs(res) <= self.residual_limit:
                ad_status.append("In Domain (Reliable)")
            elif h > self.warning_leverage_ and abs(res) <= self.residual_limit:
                ad_status.append("High Leverage (Structural Extrapolation)")
            else:
                ad_status.append("Out of Domain (Outlier)")

        return pd.DataFrame({
            "y_true": y_test,
            "y_pred": np.round(y_test_pred, 3),
            "leverage": np.round(leverages, 4),
            "std_residual": np.round(std_residuals, 3),
            "within_ad": within_ad,
            "domain_classification": ad_status,
        })
