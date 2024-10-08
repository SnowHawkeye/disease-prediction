import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from models.mipha.utils.data_processing import mask_train_test_split, scale_time_series_data_train, \
    scale_time_series_data_test


def test_mask_train_test_split():
    # Test with a small dataset
    mask = [
        (1, '2023-01-01'),
        (1, '2023-01-02'),
        (2, '2023-01-01'),
        (3, '2023-01-01'),
        (3, '2023-01-02'),
        (3, '2023-01-03'),
    ]

    train_set, test_set = mask_train_test_split(mask, test_size=0.5, random_seed=42)

    # Verify that all tuples with the same patient_id are in the same set
    train_patient_ids = {patient_id for patient_id, _ in train_set}
    test_patient_ids = {patient_id for patient_id, _ in test_set}

    assert train_patient_ids.isdisjoint(test_patient_ids), "Patient IDs should not overlap between train and test sets"

    # Verify that we have the correct number of unique patients
    assert len(train_patient_ids) + len(test_patient_ids) == 3  # There are 3 unique patients


def test_varying_test_size():
    # Test with a varying test size
    mask = [
        (1, '2023-01-01'),
        (2, '2023-01-01'),
        (3, '2023-01-01'),
        (4, '2023-01-01'),
        (5, '2023-01-01'),
    ]

    train_set_1, test_set_1 = mask_train_test_split(mask, test_size=0.2)
    train_set_2, test_set_2 = mask_train_test_split(mask, test_size=0.4)

    assert len(test_set_1) == 1  # 20% of 5 patients = 1 patient
    assert len(test_set_2) == 2  # 40% of 5 patients = 2 patients


def test_scale_time_series_data_train():
    # Generate sample 3D training data (n_samples, n_timesteps, n_features)
    train_data = np.array([
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]],
        [[9, 10], [11, 12]]
    ])  # Shape (3, 2, 2)

    scaler = StandardScaler()

    # Scale training data
    scaled_train_data = scale_time_series_data_train(train_data, scaler)

    # Check the shape of the scaled data
    assert scaled_train_data.shape == train_data.shape

    # Check that the scaler has been fitted and applied
    assert not np.allclose(train_data, scaled_train_data)


def test_scale_time_series_data_test():
    # Generate sample 3D training and test data (n_samples, n_timesteps, n_features)
    train_data = np.array([
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]],
        [[9, 10], [11, 12]]
    ])  # Shape (3, 2, 2)

    test_data = np.array([
        [[13, 14], [15, 16]],
        [[17, 18], [19, 20]]
    ])  # Shape (2, 2, 2)

    scaler = StandardScaler()

    # Fit the scaler to the training data
    scale_time_series_data_train(train_data, scaler)

    # Scale test data
    scaled_test_data = scale_time_series_data_test(test_data, scaler)

    # Check the shape of the scaled test data
    assert scaled_test_data.shape == test_data.shape

    # Check that the test data is different after scaling
    assert not np.allclose(test_data, scaled_test_data)


def test_scaler_consistency():
    # Ensure the same scaler is applied consistently to both train and test data
    np.random.seed(42)
    train_data = np.random.rand(10, 5, 3)  # Shape (10, 5, 3)
    test_data = np.random.rand(5, 5, 3)  # Shape (5, 5, 3)

    scaler = StandardScaler()

    # Scale the training data
    scaled_train_data = scale_time_series_data_train(train_data, scaler)

    # Scale the test data using the same scaler
    scaled_test_data = scale_time_series_data_test(test_data, scaler)

    # Check that both sets have been transformed by the same scaler
    assert scaled_train_data.mean() < 1e-5  # Train set should be centered around 0 (not necessarily true for test set)
