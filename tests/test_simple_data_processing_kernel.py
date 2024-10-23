from unittest.mock import MagicMock

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from models.mipha.components.kernels.simple_data_processing_kernel import SimpleDataProcessingKernel


class DummyModel:
    def fit(self):
        pass

    def predict(self):
        pass

    def predict_proba(self):
        pass


@pytest.fixture
def simple_data_processing_kernel():
    # Fixture for the SimpleDataProcessingKernel instance
    return SimpleDataProcessingKernel(model=DummyModel)


def test_fit(simple_data_processing_kernel):
    x_train = np.array([[1, 2], [3, 4], [5, 6]])
    y_train = np.array([0, 1, 0])

    # Mock the model's fit method
    simple_data_processing_kernel.model.fit = MagicMock()
    simple_data_processing_kernel.fit(x_train, y_train)

    # Verify that the process_train_data and model's fit were called with the correct data
    simple_data_processing_kernel.model.fit.assert_called_once()


def test_predict(simple_data_processing_kernel):
    # Mock input data as numpy arrays
    x_test = np.array([[1, 2], [3, 4], [5, 6]])

    # Mock the process_test_data method

    # Mock the model's predict method
    simple_data_processing_kernel.model.predict = MagicMock(return_value=np.array([0, 1, 0]))

    # Call the predict method
    result = simple_data_processing_kernel.predict(x_test)

    # Verify that process_test_data and model's predict were called with the correct data
    simple_data_processing_kernel.model.predict.assert_called_once()

    # Check the result
    np.testing.assert_array_equal(result, np.array([0, 1, 0]))


def test_predict_proba(simple_data_processing_kernel):
    # Mock input data as numpy arrays
    x_test = np.array([[1, 2], [3, 4], [5, 6]])

    # Mock the model's predict_proba method
    simple_data_processing_kernel.model.predict_proba = MagicMock(
        return_value=np.array([[0.7, 0.3], [0.4, 0.6], [0.8, 0.2]]))

    # Call the predict_proba method
    result = simple_data_processing_kernel.predict_proba(x_test)

    # Verify that process_test_data and model's predict_proba were called with the correct data
    simple_data_processing_kernel.model.predict_proba.assert_called_once()

    # Check the result
    np.testing.assert_array_equal(result, np.array([[0.7, 0.3], [0.4, 0.6], [0.8, 0.2]]))
