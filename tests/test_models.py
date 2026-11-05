import pytest
import numpy as np
from src.models import QSARModelSuite
from src.evaluate import calculate_metrics


def test_models_train_and_predict():
    np.random.seed(42)
    X_train = np.random.randn(30, 5)
    y_train = 5.0 + 1.5 * X_train[:, 0] - 0.8 * X_train[:, 1] + np.random.randn(30) * 0.1

    suite = QSARModelSuite(random_state=42)
    suite.train_all(X_train, y_train)

    for name, model in suite.models.items():
        preds = model.predict(X_train)
        metrics = calculate_metrics(y_train, preds)
        assert metrics["r2"] > 0.40
        assert metrics["rmse"] < 1.5
