import numpy as np
import pandas as pd
import pytest

from models.mipha.data_sources.mimic.patient_record_datasource import PatientRecordDataSource


@pytest.fixture
def sample_patient_records():
    """Mock data for patient records as dictionaries containing DataFrames."""
    np.random.seed(42)
    return {
        'patient_1': {
            '2021-01-01': pd.DataFrame(np.random.rand(10, 5)),
            '2021-01-02': pd.DataFrame(np.random.rand(10, 5))
        },
        'patient_2': {
            '2021-01-01': pd.DataFrame(np.random.rand(10, 5)),
            '2021-01-02': pd.DataFrame(np.random.rand(10, 5)),
            '2021-01-03': pd.DataFrame(np.random.rand(10, 5)),
        }
    }


@pytest.fixture
def sample_labels():
    return np.array([0, 1, 1, 0, 1])


@pytest.fixture
def sample_numpy_data():
    """Mock pre-formatted 3D numpy array data."""
    return np.random.rand(10, 2, 5)


@pytest.fixture
def mock_mask():
    """Mock mask as a list of (patient_id, timestamp) tuples."""
    return [('patient_1', '2021-01-01'), ('patient_1', '2021-01-02')]


@pytest.fixture
def datasource_with_dict(sample_patient_records):
    """Fixture for `PatientRecordDatasource` initialized with a dictionary."""
    return PatientRecordDataSource(data_type="dict", name="sample_dict", data=sample_patient_records)


@pytest.fixture
def datasource_with_numpy(sample_numpy_data):
    """Fixture for `PatientRecordDatasource` initialized with a numpy array."""
    return PatientRecordDataSource(data_type="numpy", name="sample_numpy", data=sample_numpy_data)


def test_datasource_initialization_dict(datasource_with_dict):
    """Test that data from a dictionary is converted into a numpy array and mask is created."""
    assert isinstance(datasource_with_dict.data, np.ndarray), "Data should be converted to a numpy array."
    assert datasource_with_dict.mask is not None, "Mask should be created from the patient records."


def test_datasource_initialization_numpy(datasource_with_numpy):
    """Test that data as a numpy array is correctly set and mask is None."""
    assert isinstance(datasource_with_numpy.data, np.ndarray), "Data should be a numpy array."
    assert datasource_with_numpy.mask is None, "Mask should be None for pre-formatted numpy array."


def test_warning_for_missing_mask(sample_numpy_data):
    """Test that a warning is issued if no mask is provided for numpy data."""
    with pytest.warns(UserWarning, match="Mask could not be created"):
        PatientRecordDataSource(data_type="numpy", name="sample_numpy", data=sample_numpy_data)


def test_split_train_test_without_scaler(datasource_with_dict, sample_labels):
    """Test the split_train_test method with mocked mask splitting."""
    train_datasource, test_datasource, train_labels, test_labels = \
        datasource_with_dict.split_train_test(random_seed=0, labels=sample_labels)

    assert isinstance(train_datasource, PatientRecordDataSource), "Train datasource should be PatientRecordDatasource."
    assert isinstance(test_datasource, PatientRecordDataSource), "Test datasource should be PatientRecordDatasource."
    assert len(train_datasource.data) == 2, "Train data should contain 2 records (patient 1)."
    assert len(test_datasource.data) == 3, "Test data should contain 3 records (patient 2)."
