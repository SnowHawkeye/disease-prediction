import pytest
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from models.mipha.components.kernels.random_forest_kernel import RandomForestKernel


@pytest.fixture
def data():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def test_rf_kernel_fit_predict(data):
    X_train, X_test, y_train, y_test = data
    rf_kernel = RandomForestKernel(component_name='rf_kernel_test', n_estimators=10, random_state=42)

    # Fit the model
    rf_kernel.fit(X_train, y_train)

    # Make predictions
    y_pred = rf_kernel.predict(X_test)

    assert len(y_pred) == len(y_test), "Prediction length should match test set length"
    assert np.unique(y_pred).shape[0] <= 3, "Should predict 3 or fewer unique classes for the iris dataset"


def test_rf_kernel_predict_proba(data):
    X_train, X_test, y_train, y_test = data
    rf_kernel = RandomForestKernel(component_name='rf_kernel_test', n_estimators=10, random_state=42)

    # Fit the model
    rf_kernel.fit(X_train, y_train)

    # Predict probabilities
    y_proba = rf_kernel.predict_proba(X_test)

    assert y_proba.shape == (X_test.shape[0], 3), "Predicted probabilities shape should be (n_samples, n_classes)"
    assert np.allclose(np.sum(y_proba, axis=1), 1), "Probabilities for each sample should sum to 1"
