import pytest
import numpy as np
from src.applicability_domain import ApplicabilityDomain


def test_applicability_domain_leverage():
    np.random.seed(42)
    X_train = np.random.randn(40, 4)
    y_train = np.random.randn(40)
    y_train_pred = y_train + np.random.randn(40) * 0.1

    ad = ApplicabilityDomain()
    ad.fit(X_train, y_train, y_train_pred)

    assert ad.warning_leverage_ > 0.0

    # Normal in-distribution test sample
    X_test_normal = np.random.randn(5, 4) * 0.5
    y_test_normal = np.random.randn(5)
    y_test_pred = y_test_normal

    res_df = ad.evaluate_test_set(X_test_normal, y_test_normal, y_test_pred)
    assert len(res_df) == 5
    assert "leverage" in res_df.columns
    assert "within_ad" in res_df.columns
