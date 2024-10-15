import pytest
import numpy as np

from models.mipha.data_sources.mimic.demographics_datasource import DemographicsDataSource


@pytest.fixture
def sample_data():
    # Create a simple dictionary to simulate demographic data
    return {
        'patient_1': {
            '2024-10-01': {'age': 65, 'gender': 'M'},
            '2024-10-02': {'age': 65, 'gender': 'M'}
        },
        'patient_2': {
            '2024-10-01': {'age': 72, 'gender': 'F'}
        }
    }


@pytest.fixture
def mock_mask():
    return [('patient_1', '2024-10-01'), ('patient_2', '2024-10-01')]


def test_init_without_mask(sample_data):
    # GIVEN
    expected_mask = [('patient_1', '2024-10-01'), ('patient_1', '2024-10-02'), ('patient_2', '2024-10-01')]
    expected_data = np.array([
        [[65, 1]],
        [[65, 1]],
        [[72, 0]],
    ], dtype=object)  # if not set manually, numpy interprets the mixture of str and int as str

    # WHEN
    ds = DemographicsDataSource(data_type='demographics', name='test_ds', data=sample_data)

    # THEN
    assert ds.mask == expected_mask
    np.testing.assert_array_equal(ds.data, expected_data)


def test_init_with_mask(sample_data, mock_mask):
    # GIVEN
    expected_data = np.array([
        [[65, 1]],
        [[72, 0]],
    ], dtype=object)  # if not set manually, numpy interprets the mixture of str and int as str

    # WHEN
    ds = DemographicsDataSource(data_type='demographics', name='test_ds', data=sample_data, mask=mock_mask)

    # THEN
    assert ds.mask == mock_mask
    np.testing.assert_array_equal(ds.data, expected_data)


def test_extend_timesteps_2d_data(sample_data, mock_mask):
    # GIVEN
    input_2d = np.array([
        [65, 1],
        [72, 0]
    ], dtype=object)

    n_timesteps = 5
    expected_output = np.array([
        [[65, 1]] * n_timesteps,
        [[72, 0]] * n_timesteps,
    ], dtype=object)

    # WHEN
    ds = DemographicsDataSource(data_type='demographics', name='test_ds', data=sample_data, mask=mock_mask)
    ds.data = input_2d  # we "mock" the flatten function
    output = ds.extend_timesteps(n_timesteps=n_timesteps)

    # THEN
    np.testing.assert_array_equal(output, expected_output)
    np.testing.assert_array_equal(ds.data, expected_output)


def test_extend_timesteps_3d_single_timestep(sample_data, mock_mask):
    # GIVEN
    n_timesteps = 5
    expected_output = np.array([
        [[65, 1]] * n_timesteps,
        [[72, 0]] * n_timesteps,
    ], dtype=object)

    # WHEN
    ds = DemographicsDataSource(data_type='demographics', name='test_ds', data=sample_data, mask=mock_mask)
    output = ds.extend_timesteps(n_timesteps=n_timesteps)

    # THEN
    np.testing.assert_array_equal(output, expected_output)
    np.testing.assert_array_equal(ds.data, expected_output)


def test_extend_timesteps_3d_multiple_timesteps(sample_data, mock_mask):
    # GIVEN
    n_timesteps = 5
    input_3d = np.array([
        [[65, 1]] * n_timesteps,
        [[72, 0]] * n_timesteps,
    ], dtype=object)

    # WHEN
    ds = DemographicsDataSource(data_type='demographics', name='test_ds', data=sample_data, mask=mock_mask)
    ds.data = input_3d  # "mocking" 3d data that was already reshaped
    output = ds.extend_timesteps(n_timesteps=n_timesteps)

    # THEN

    with pytest.warns(UserWarning, match="Data shape .* does not require reshaping"):
        output = ds.extend_timesteps(n_timesteps=5)

    np.testing.assert_array_equal(ds.data, input_3d)
    np.testing.assert_array_equal(output, input_3d)


def test_flatten_data(sample_data, mock_mask):
    # GIVEN
    expected_output = np.array([
        [65, 1],
        [72, 0]
    ], dtype=object)

    # WHEN
    ds = DemographicsDataSource(data_type='demographics', name='test_ds', data=sample_data, mask=mock_mask)
    output = ds.flatten_data()

    # THEN
    np.testing.assert_array_equal(output, expected_output)
