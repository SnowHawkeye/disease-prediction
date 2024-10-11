from unittest.mock import MagicMock

import numpy as np

from models.mipha.components.kernels.data_processing_kernel import DataProcessingKernel


class MockImplementation(DataProcessingKernel):
    def fit(self, x_train, y_train, *args, **kwargs):
        pass

    def predict(self, x_test, *args, **kwargs):
        pass

    def __init__(self, imputer=None, resampler=None, scaler=None):
        super().__init__(imputer=imputer, resampler=resampler, scaler=scaler)

    def process_train_data(self, x_train, y_train):
        return super().process_train_data(x_train, y_train)

    def process_test_data(self, x_test):
        return super().process_test_data(x_test)


def test_process_train_data_with_imputer_scaler():
    # Arrange
    model = MockImplementation(
        imputer=MagicMock(),
        resampler=MagicMock(),
        scaler=MagicMock(),
    )
    x_train = np.array([[1, 2], [3, 4]])
    y_train = np.array([0, 1])

    model.imputer.fit_transform.return_value = np.array([[1, 2], [3, 4]])
    model.resampler.fit_resample.return_value = (np.array([[1, 2], [3, 4]]), np.array([0, 1]))
    model.scaler.fit_transform.return_value = np.array([[0, 1], [1, 0]])

    # Act
    processed_x, processed_y = model.process_train_data(x_train, y_train)

    # Assert
    assert np.array_equal(processed_x, np.array([[0, 1], [1, 0]]))
    assert np.array_equal(processed_y, y_train)
    model.imputer.fit_transform.assert_called_once()
    model.resampler.fit_resample.assert_called_once()
    model.scaler.fit_transform.assert_called_once()


def test_process_train_data_with_only_imputer():
    # Arrange
    model = MockImplementation(
        imputer=MagicMock(),
    )
    x_train = np.array([[1, 2], [3, 4]])
    y_train = np.array([0, 1])

    model.imputer.fit_transform.return_value = np.array([[1, 2], [3, 4]])

    # Act
    processed_x, processed_y = model.process_train_data(x_train, y_train)

    # Assert
    assert np.array_equal(processed_x, np.array([[1, 2], [3, 4]]))
    assert np.array_equal(processed_y, y_train)
    model.imputer.fit_transform.assert_called_once()


def test_process_train_data_with_only_resampler():
    # Arrange
    model = MockImplementation(
        resampler=MagicMock(),
    )
    x_train = np.array([[1, 2], [3, 4]])
    y_train = np.array([0, 1])

    model.resampler.fit_resample.return_value = (np.array([[1, 2], [3, 4]]), np.array([0, 1]))

    # Act
    processed_x, processed_y = model.process_train_data(x_train, y_train)

    # Assert
    assert np.array_equal(processed_x, np.array([[1, 2], [3, 4]]))
    assert np.array_equal(processed_y, y_train)
    model.resampler.fit_resample.assert_called_once()


def test_process_train_data_with_no_components():
    # Arrange
    model = MockImplementation()
    x_train = np.array([[1, 2], [3, 4]])
    y_train = np.array([0, 1])

    # Act
    processed_x, processed_y = model.process_train_data(x_train, y_train)

    # Assert
    assert np.array_equal(processed_x, x_train)  # Expect no change
    assert np.array_equal(processed_y, y_train)  # Expect no change


def test_process_train_data_with_all_none_components():
    # Arrange
    model = MockImplementation()
    x_train = np.array([[1, 2], [3, 4]])
    y_train = np.array([0, 1])

    # Act
    processed_x, processed_y = model.process_train_data(x_train, y_train)

    # Assert
    assert np.array_equal(processed_x, x_train)  # Expect no change
    assert np.array_equal(processed_y, y_train)  # Expect no change


def test_process_train_data_with_3d_data_and_only_scaler():
    # Arrange
    model = MockImplementation(
        scaler=MagicMock(),
    )
    model.n_dim = 3  # Set the dimensionality to 3
    x_train = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    y_train = np.array([0, 1])

    model.scaler.fit_transform.return_value = np.array([[[0, 1], [1, 0]], [[1, 0], [0, 1]]])

    # Act
    processed_x, processed_y = model.process_train_data(x_train, y_train)

    # Assert
    assert np.array_equal(processed_x, np.array([[[0, 1], [1, 0]], [[1, 0], [0, 1]]]))
    assert np.array_equal(processed_y, y_train)
    model.scaler.fit_transform.assert_called_once()


def test_process_train_data_with_3d_data_and_only_imputer():
    # Arrange
    model = MockImplementation(
        imputer=MagicMock(),
    )
    model.n_dim = 3  # Set the dimensionality to 3
    x_train = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    y_train = np.array([0, 1])

    model.imputer.fit_transform.side_effect = lambda x: x

    # Act
    processed_x, processed_y = model.process_train_data(x_train, y_train)

    # Assert
    assert np.array_equal(processed_x, x_train)
    assert np.array_equal(processed_y, y_train)
    model.imputer.fit_transform.assert_called()


def test_process_train_data_with_3d_data_and_both_imputer_and_scaler():
    # Arrange
    model = MockImplementation(
        imputer=MagicMock(),
        scaler=MagicMock(),
    )
    model.n_dim = 3  # Set the dimensionality to 3
    x_train = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    y_train = np.array([0, 1])

    model.imputer.fit_transform.side_effect = lambda x: x
    model.scaler.fit_transform.return_value = np.array([[[0, 1], [1, 0]], [[1, 0], [0, 1]]])

    # Act
    processed_x, processed_y = model.process_train_data(x_train, y_train)

    # Assert
    assert np.array_equal(processed_x, np.array([[[0, 1], [1, 0]], [[1, 0], [0, 1]]]))
    assert np.array_equal(processed_y, y_train)
    model.imputer.fit_transform.assert_called()
    model.scaler.fit_transform.assert_called_once()


def test_process_test_data_with_imputer_scaler():
    # Arrange
    model = MockImplementation(
        imputer=MagicMock(),
        scaler=MagicMock(),
    )
    x_test = np.array([[1, 2], [3, 4]])

    model.imputer.transform.return_value = np.array([[1, 2], [3, 4]])
    model.scaler.transform.return_value = np.array([[0, 1], [1, 0]])

    # Act
    processed_x = model.process_test_data(x_test)

    # Assert
    assert np.array_equal(processed_x, np.array([[0, 1], [1, 0]]))
    model.imputer.transform.assert_called_once()
    model.scaler.transform.assert_called_once()


def test_process_test_data_with_only_scaler():
    # Arrange
    model = MockImplementation(
        scaler=MagicMock(),
    )
    x_test = np.array([[1, 2], [3, 4]])

    model.scaler.transform.return_value = np.array([[0, 1], [1, 0]])

    # Act
    processed_x = model.process_test_data(x_test)

    # Assert
    assert np.array_equal(processed_x, np.array([[0, 1], [1, 0]]))
    model.scaler.transform.assert_called_once()


def test_process_test_data_with_only_imputer():
    # Arrange
    model = MockImplementation(
        imputer=MagicMock(),
    )
    x_test = np.array([[1, 2], [3, 4]])

    model.imputer.transform.return_value = np.array([[1, 2], [3, 4]])

    # Act
    processed_x = model.process_test_data(x_test)

    # Assert
    assert np.array_equal(processed_x, np.array([[1, 2], [3, 4]]))
    model.imputer.transform.assert_called_once()


def test_process_test_data_with_only_resampler():
    # Arrange
    model = MockImplementation(
        resampler=MagicMock(),
    )
    x_test = np.array([[1, 2], [3, 4]])

    # There shouldn't be any transformation since resampler is for training only
    processed_x = model.process_test_data(x_test)

    # Assert
    assert np.array_equal(processed_x, x_test)  # Expect no change
    assert not model.resampler.fit_resample.called  # Resampler shouldn't be called


def test_process_test_data_with_no_components():
    # Arrange
    model = MockImplementation()
    x_test = np.array([[1, 2], [3, 4]])

    # Act
    processed_x = model.process_test_data(x_test)

    # Assert
    assert np.array_equal(processed_x, x_test)  # Expect no change


def test_process_test_data_with_3d_data():
    # Arrange
    model = MockImplementation(
        # because of the way 3D imputation is managed, mocking the imputer is difficult
        imputer=None,
        scaler=MagicMock(),
    )
    model.n_dim = 3  # Set the dimensionality to 3
    x_test = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

    model.scaler.transform.return_value = np.array([[[0, 1], [1, 0]], [[1, 0], [0, 1]]])

    # Act
    model.process_test_data(x_test)

    # Assert
    # not testing the output because the imputer
    model.scaler.transform.assert_called_once()
