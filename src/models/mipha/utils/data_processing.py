from sklearn.impute import SimpleImputer
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split


def mask_train_test_split(mask, test_size=0.2, random_seed=None):
    """
    Given a list of (patient_id, timestamp) tuples, return a train-test split of the data.
    The split is based on patients, meaning that two tuples with the same patient ID will always be in the same set.

    :param mask: A list of (patient_id, timestamp) tuples.
    :type mask: list[tuple]

    :param test_size: Proportion of patient IDs to include in the test split (default is 0.2).
    :type test_size: float

    :param random_seed: Seed for the random number generator for reproducibility (default is None).
    :type random_seed: int or None

    :return: A tuple containing two lists:
        - train_set: A list of (patient_id, timestamp) tuples for training.
        - test_set: A list of (patient_id, timestamp) tuples for testing.
    :rtype: tuple(list[tuple], list[tuple])
    """

    # Extract unique patient IDs and group by patient
    patient_groups = {}
    for patient_id, timestamp in mask:
        if patient_id not in patient_groups:
            patient_groups[patient_id] = []
        patient_groups[patient_id].append((patient_id, timestamp))

    # Extract unique patient IDs
    unique_patient_ids = list(patient_groups.keys())

    # Split patient IDs into train and test sets using scikit-learn
    train_patient_ids, test_patient_ids = train_test_split(
        unique_patient_ids, test_size=test_size, random_state=random_seed
    )

    # Create train and test sets based on patient IDs
    train_set = [item for patient_id in train_patient_ids for item in patient_groups[patient_id]]
    test_set = [item for patient_id in test_patient_ids for item in patient_groups[patient_id]]

    return train_set, test_set


def make_simple_imputer(strategy="mean"):
    return SimpleImputer(keep_empty_features=True, strategy=strategy)


def impute_data(data, imputer):
    """
    Performs data imputation on a per-matrix basis.
    :param data: iterable of history matrices
    :param imputer: the imputer to use to replace missing data
    :return: a list of history matrices with imputed data
    """
    print("Imputing data...")
    data_imputed = []
    for matrix in tqdm(data):
        matrix_imputed = imputer.fit_transform(matrix)
        data_imputed.append(matrix_imputed)
    print("Data successfully imputed!")
    return np.array(data_imputed)


def scale_time_series_data_train(train_data, scaler):
    """
    Fit the scaler to the provided training data, and return the scaled training data.
    :param train_data: training data, of shape (n_samples, n_timesteps, n_features)
    :param scaler: scaler to fit and apply to the data
    """
    x_train_shape = train_data.shape
    n_samples, n_timesteps, n_features = x_train_shape

    print("Scaling training data...")

    x_train_reshape = np.reshape(train_data, newshape=(n_samples, n_timesteps * n_features))

    x_train_reshape = scaler.fit_transform(x_train_reshape)
    x_train_scaled = x_train_reshape.reshape(x_train_shape)

    print("Training data scaled successfully!")

    return x_train_scaled


def scale_time_series_data_test(test_data, trained_scaler):
    """
    Use the given scaler to scale the provided test data, and return the scaled test data.
    :param test_data: training data, of shape (n_samples, n_timesteps, n_features)
    :param trained_scaler: trained scaler to use
    """
    x_test_shape = test_data.shape
    n_samples, n_timesteps, n_features = x_test_shape

    print("Scaling test data...")

    x_test_reshape = np.reshape(test_data, newshape=(n_samples, n_timesteps * n_features))

    x_test_reshape = trained_scaler.transform(x_test_reshape)
    x_test_scaled = x_test_reshape.reshape(x_test_shape)

    print("Test data scaled successfully!")

    return x_test_scaled


def resample_3d_data(data, labels, resampler):
    """
    Resamples imbalanced 3D time series data using a provided resampler.

    This function reshapes the 3D time series data into 2D, applies the given resampler
    (e.g., RandomOverSampler) to handle class imbalances, and then reshapes the data back to 3D.

    :param data: 3D numpy array of shape (n_samples, n_timesteps, n_features) representing the time series data.
    :type data: np.ndarray
    :param labels: 1D numpy array of shape (n_samples,) representing the labels associated with each sample.
    :type labels: np.ndarray
    :param resampler: A resampling object (e.g., RandomOverSampler) from imbalanced-learn or similar library,
                      which has a `fit_resample(X, y)` method.

    :return: Tuple containing the resampled 3D data and resampled labels.
    :rtype: tuple (np.ndarray, np.ndarray)

    :raises ValueError: If the input `data` is not 3D or if `labels` do not match the number of samples in `data`.
    """

    # Check if input data is 3D
    if data.ndim != 3:
        raise ValueError("Input data must be 3D, with shape (n_samples, n_timesteps, n_features).")

    # Check if the number of labels matches the number of samples
    n_samples, n_timesteps, n_features = data.shape
    if len(labels) != n_samples:
        raise ValueError("Number of labels must match the number of samples in data.")

    # Reshape the 3D data to 2D for resampling
    reshaped_data = data.reshape(
        (n_samples, n_timesteps * n_features))  # Flatten into (n_samples, n_timesteps * n_features)

    # Apply resampling
    resampled_data_2d, resampled_labels = resampler.fit_resample(reshaped_data, labels)

    # Reshape the resampled data back to 3D
    resampled_n_samples = resampled_data_2d.shape[0]
    resampled_data_3d = resampled_data_2d.reshape((resampled_n_samples, n_timesteps, n_features))

    return resampled_data_3d, resampled_labels
