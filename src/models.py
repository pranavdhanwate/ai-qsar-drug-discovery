from typing import Dict, Any, Tuple
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
import xgboost as xgb


class QSARModelSuite:
    """Manages baseline Ridge, Random Forest, and XGBoost QSAR regressor models."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models: Dict[str, Any] = {
            "Ridge Regression": Ridge(alpha=1.0),
            "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=8, random_state=random_state),
            "XGBoost": xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.08, random_state=random_state),
        }
        self.best_params_: Dict[str, Dict[str, Any]] = {}

    def tune_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Optimizes hyperparameters for Random Forest and XGBoost using cross-validation."""
        cv = KFold(n_splits=5, shuffle=True, random_state=self.random_state)

        # Random Forest Grid
        rf_grid = {
            "n_estimators": [50, 100, 150],
            "max_depth": [4, 6, 8, None],
            "min_samples_split": [2, 4],
        }
        rf_search = GridSearchCV(
            RandomForestRegressor(random_state=self.random_state),
            rf_grid,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )
        rf_search.fit(X_train, y_train)
        self.models["Random Forest"] = rf_search.best_estimator_
        self.best_params_["Random Forest"] = rf_search.best_params_

        # XGBoost Grid
        xgb_grid = {
            "n_estimators": [50, 100],
            "max_depth": [3, 4, 6],
            "learning_rate": [0.03, 0.08, 0.15],
        }
        xgb_search = GridSearchCV(
            xgb.XGBRegressor(random_state=self.random_state),
            xgb_grid,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )
        xgb_search.fit(X_train, y_train)
        self.models["XGBoost"] = xgb_search.best_estimator_
        self.best_params_["XGBoost"] = xgb_search.best_params_

    def train_all(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fits all candidate models on the training partition."""
        for name, model in self.models.items():
            model.fit(X_train, y_train)

    def evaluate_cv(self, X_train: np.ndarray, y_train: np.ndarray, n_splits: int = 5) -> Dict[str, Dict[str, float]]:
        """Calculates 5-fold cross-validated R² and RMSE on training data."""
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        cv_results = {}

        for name, model in self.models.items():
            r2_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2")
            rmse_scores = -cross_val_score(model, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error")

            cv_results[name] = {
                "cv_r2_mean": float(np.mean(r2_scores)),
                "cv_r2_std": float(np.std(r2_scores)),
                "cv_rmse_mean": float(np.mean(rmse_scores)),
                "cv_rmse_std": float(np.std(rmse_scores)),
            }
        return cv_results
