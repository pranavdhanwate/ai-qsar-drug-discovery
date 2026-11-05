from typing import List, Tuple
import numpy as np
from sklearn.feature_selection import VarianceThreshold


class FeatureSelector:
    """Removes low-variance and highly collinear redundant features."""

    def __init__(self, variance_threshold: float = 0.01, correlation_threshold: float = 0.95):
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.var_selector = VarianceThreshold(threshold=variance_threshold)
        self.retained_indices: List[int] = []

    def fit_transform(self, X: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        # 1. Variance threshold
        X_var = self.var_selector.fit_transform(X)
        support = self.var_selector.get_support(indices=True)
        names_var = [feature_names[i] for i in support]

        # 2. Pairwise correlation filtering
        corr_matrix = np.corrcoef(X_var, rowvar=False)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        n_features = X_var.shape[1]
        to_drop = set()

        for i in range(n_features):
            if i in to_drop:
                continue
            for j in range(i + 1, n_features):
                if j in to_drop:
                    continue
                if abs(corr_matrix[i, j]) > self.correlation_threshold:
                    to_drop.add(j)

        keep_indices = [idx for idx in range(n_features) if idx not in to_drop]
        self.retained_indices = keep_indices

        X_clean = X_var[:, keep_indices]
        names_clean = [names_var[idx] for idx in keep_indices]
        return X_clean, names_clean

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_var = self.var_selector.transform(X)
        return X_var[:, self.retained_indices]
