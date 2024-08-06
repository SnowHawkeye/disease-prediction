import os
import pickle
import tempfile
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from datetime import datetime

from src.features.mimic.extract_lab_records import extract_lab_records, save_patient_records_to_parquet_archive, \
    load_patient_records_from_pickle, \
    save_patient_records_to_pickle, load_patient_records_from_parquet_archive, filter_lab_records, make_rolling_records, \
    label_lab_records


@pytest.fixture
def mock_data():
    # Mock data setup
    data = {
        'patient_id': [1, 1, 2, 2, 3, 3, None],
        'analysis_time': ['2022-01-01 10:00', '2022-01-01 11:00', '2022-01-02 10:00', None, '2022-01-03 10:00',
                          '2022-01-03 11:00', '2022-01-04 10:00'],
        'analysis_id': ['A', 'B', 'A', 'B', 'A', 'B', 'A'],
        'analysis_value': [10, 15, 20, 25, 30, 35, 40]
    }
    df = pd.DataFrame(data)
    df['analysis_time'] = pd.to_datetime(df['analysis_time'], errors='coerce')  # Convert to datetime
    return df


def test_extract_biological_data(mock_data):
    result = extract_lab_records(mock_data)

    # Expected results for patient 1 and patient 3 (patient 2 has NaN analysis_time and should be ignored)
    expected_patient_1 = pd.DataFrame({
        'A': [10, None],
        'B': [None, 15]
    }, index=pd.to_datetime(['2022-01-01 10:00', '2022-01-01 11:00']))
    expected_patient_1.index.name = 'analysis_time'
    expected_patient_1.rename_axis('analysis_id', axis='columns', inplace=True)

    expected_patient_2 = pd.DataFrame({
        'A': [20],
    }, index=pd.to_datetime(['2022-01-02 10:00']))

    expected_patient_3 = pd.DataFrame({
        'A': [30, None],
        'B': [None, 35]
    }, index=pd.to_datetime(['2022-01-03 10:00', '2022-01-03 11:00']))
    expected_patient_3.index.name = 'analysis_time'
    expected_patient_3.rename_axis('analysis_id', axis='columns', inplace=True)

    assert 1 in result
    assert result[1].to_numpy().all() == expected_patient_1.to_numpy().all()

    assert 3 in result
    assert result[3].to_numpy().all() == expected_patient_3.to_numpy().all()

    assert 2 in result
    assert result[2].to_numpy().all() == expected_patient_2.to_numpy().all()


# Filtering
# Sample DataFrames for testing
patient_lab_records = {
    'patient1': pd.DataFrame({'A': [1, 2], 'B': [3, 4], 'C': [5, 6]}),
    'patient2': pd.DataFrame({'A': [7, 8], 'D': [9, 10]}),
    'patient3': pd.DataFrame({'B': [11, 12], 'C': [13, 14]})
}

analyses_ids = ['A', 'B', 'E']


def test_filter_lab_records_basic_functionality():
    filtered = filter_lab_records(patient_lab_records, analyses_ids)
    assert 'A' in filtered['patient1'].columns
    assert 'B' in filtered['patient1'].columns
    assert 'E' in filtered['patient1'].columns
    assert pd.isna(filtered['patient1']['E']).all()  # 'E' should be NaN-filled


def test_filter_lab_records_missing_columns():
    filtered = filter_lab_records(patient_lab_records, analyses_ids)
    assert pd.isna(filtered['patient2']['E']).all()
    assert pd.isna(filtered['patient3']['A']).all()


def test_filter_lab_records_no_common_columns():
    filtered = filter_lab_records(patient_lab_records, ['X', 'Y'])
    for df in filtered.values():
        assert pd.isna(df).all().all()  # All values should be NaN


def test_filter_lab_records_empty_input():
    empty_records = {}
    filtered = filter_lab_records(empty_records, analyses_ids)
    assert filtered == {}


def test_filter_lab_records_mixed_presence():
    filtered = filter_lab_records(patient_lab_records, ['B', 'D', 'E'])
    assert 'B' in filtered['patient1'].columns
    assert 'D' in filtered['patient1'].columns  # Should be NaN-filled
    assert pd.isna(filtered['patient1']['D']).all()
    assert 'E' in filtered['patient2'].columns  # Should be NaN-filled
    assert pd.isna(filtered['patient2']['E']).all()


# Making rolling records

def test_make_rolling_records_day():
    # Test with time_unit='day' and backward_window=3
    data = {
        'patient_1': pd.DataFrame({
            'value': [1, 2, 3, 4],
        }, index=pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'])),
        'patient_2': pd.DataFrame({
            'value': [1, 2, 3, 4],
        }, index=pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04']))
    }
    expected = {
        'patient_1': {
            pd.Timestamp('2023-01-01'): pd.DataFrame({'value': [np.nan, np.nan, 1]},
                                                     index=pd.to_datetime(['2022-12-30', '2022-12-31', '2023-01-01'])),
            pd.Timestamp('2023-01-02'): pd.DataFrame({'value': [np.nan, 1, 2]},
                                                     index=pd.to_datetime(['2022-12-31', '2023-01-01', '2023-01-02'])),
            pd.Timestamp('2023-01-03'): pd.DataFrame({'value': [1, 2, 3]},
                                                     index=pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])),
            pd.Timestamp('2023-01-04'): pd.DataFrame({'value': [2, 3, 4]},
                                                     index=pd.to_datetime(['2023-01-02', '2023-01-03', '2023-01-04'])),

        },
        'patient_2': {
            pd.Timestamp('2023-01-01'): pd.DataFrame({'value': [np.nan, np.nan, 1]},
                                                     index=pd.to_datetime(['2022-12-30', '2022-12-31', '2023-01-01'])),
            pd.Timestamp('2023-01-02'): pd.DataFrame({'value': [np.nan, 1, 2]},
                                                     index=pd.to_datetime(['2022-12-31', '2023-01-01', '2023-01-02'])),
            pd.Timestamp('2023-01-03'): pd.DataFrame({'value': [1, 2, 3]},
                                                     index=pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])),
            pd.Timestamp('2023-01-04'): pd.DataFrame({'value': [2, 3, 4]},
                                                     index=pd.to_datetime(['2023-01-02', '2023-01-03', '2023-01-04'])),

        }
    }
    result = make_rolling_records(data, 'day', 3)
    for patient_id in result:
        for date in result[patient_id]:
            pd.testing.assert_frame_equal(result[patient_id][date], expected[patient_id][date],
                                          check_dtype=False, check_freq=False)


def test_make_rolling_records_week():
    # Test with time_unit='week' and backward_window=2
    data = {
        'patient_1': pd.DataFrame({
            'value': [1, 2],
        }, index=pd.to_datetime(['2023-01-08', '2023-01-15']))
    }
    expected = {
        'patient_1': {
            pd.Timestamp('2023-01-08'): pd.DataFrame({'value': [np.nan, 1]},
                                                     index=pd.to_datetime(['2023-01-01', '2023-01-08'])),
            pd.Timestamp('2023-01-15'): pd.DataFrame({'value': [1, 2]},
                                                     index=pd.to_datetime(['2023-01-08', '2023-01-15'])),
        }
    }
    result = make_rolling_records(data, 'week', 2)
    for patient_id in result:
        for date in result[patient_id]:
            pd.testing.assert_frame_equal(result[patient_id][date], expected[patient_id][date],
                                          check_dtype=False, check_freq=False)


def test_make_rolling_records_month():
    # Test with time_unit='month' and backward_window=2
    data = {
        'patient_1': pd.DataFrame({
            'value': [1, 2, 3, 4],
        }, index=pd.to_datetime(['2023-01-08', '2023-02-14', '2023-04-24', '2023-08-12']))
    }
    expected = {
        'patient_1': {
            pd.Timestamp('2023-01-31'): pd.DataFrame({'value': [np.nan, 1]},
                                                     index=pd.to_datetime(['2022-12-31', '2023-01-31'])),
            pd.Timestamp('2023-02-28'): pd.DataFrame({'value': [1, 2]},
                                                     index=pd.to_datetime(['2023-01-31', '2023-02-28'])),
            pd.Timestamp('2023-04-30'): pd.DataFrame({'value': [np.nan, 3]},
                                                     index=pd.to_datetime(['2023-03-31', '2023-04-30'])),
            pd.Timestamp('2023-08-31'): pd.DataFrame({'value': [np.nan, 4]},
                                                     index=pd.to_datetime(['2023-07-31', '2023-08-31'])),

        }
    }
    result = make_rolling_records(data, 'month', 2)
    for patient_id in result:
        for date in result[patient_id]:
            pd.testing.assert_frame_equal(result[patient_id][date], expected[patient_id][date],
                                          check_dtype=False, check_freq=False)


def test_make_rolling_records_year():
    # Test with time_unit='year' and backward_window=2
    data = {
        'patient_1': pd.DataFrame({
            'value': [1, 2, 3, 4],
        }, index=pd.to_datetime(['2023-01-08', '2024-02-14', '2027-04-24', '2029-08-12']))
    }
    expected = {
        'patient_1': {
            pd.Timestamp('2023-12-31'): pd.DataFrame({'value': [np.nan, 1]},
                                                     index=pd.to_datetime(['2022-12-31', '2023-12-31'])),
            pd.Timestamp('2024-12-31'): pd.DataFrame({'value': [1, 2]},
                                                     index=pd.to_datetime(['2023-12-31', '2024-12-31'])),
            pd.Timestamp('2027-12-31'): pd.DataFrame({'value': [np.nan, 3]},
                                                     index=pd.to_datetime(['2026-12-31', '2027-12-31'])),
            pd.Timestamp('2029-12-31'): pd.DataFrame({'value': [np.nan, 4]},
                                                     index=pd.to_datetime(['2028-12-31', '2029-12-31'])),

        }
    }
    result = make_rolling_records(data, 'year', 2)
    for patient_id in result:
        for date in result[patient_id]:
            pd.testing.assert_frame_equal(result[patient_id][date], expected[patient_id][date],
                                          check_dtype=False, check_freq=False)


# Labeling function

def test_label_lab_records():
    # Sample input data
    patient_rolling_lab_records = {
        1: {
            datetime(2021, 1, 1): pd.DataFrame({"lab_result": [1, 2, 3]}),
            datetime(2021, 2, 1): pd.DataFrame({"lab_result": [4, 5, 6]})
        },
        2: {
            datetime(2021, 1, 1): pd.DataFrame({"lab_result": [7, 8, 9]})
        }
    }

    gap_days = 1
    prediction_window_days = 10
    positive_diagnoses = ['D1', 'D2']

    diagnoses_data = {
        'patient_id': [1, 1, 2],
        'admission_id': [100, 101, 200],
        'diagnosis_code': ['D1', 'D3', 'D2']
    }
    diagnoses_table = pd.DataFrame(diagnoses_data)

    admissions_data = {
        'patient_id': [1, 1, 2],
        'admission_id': [100, 101, 200],
        'discharge_time': ['2021-01-05', '2021-02-05', '2021-01-15']
    }
    admissions_table = pd.DataFrame(admissions_data)
    admissions_table["discharge_time"] = pd.to_datetime(admissions_table["discharge_time"])

    expected_output = {
        1: {
            datetime(2021, 1, 1): 1,
            datetime(2021, 2, 1): 0
        },
        2: {
            datetime(2021, 1, 1): 0
        }
    }

    # Function call
    result = label_lab_records(patient_rolling_lab_records, gap_days, prediction_window_days, positive_diagnoses,
                               diagnoses_table, admissions_table)

    # Assert
    assert result == expected_output


def test_label_lab_records_no_diagnoses():
    # Sample input data
    patient_rolling_lab_records = {
        1: {
            datetime(2021, 1, 1): pd.DataFrame({"lab_result": [1, 2, 3]}),
            datetime(2021, 2, 1): pd.DataFrame({"lab_result": [4, 5, 6]})
        },
        2: {
            datetime(2021, 1, 1): pd.DataFrame({"lab_result": [7, 8, 9]})
        }
    }

    gap_days = 1
    prediction_window_days = 10
    positive_diagnoses = ['D1', 'D2']

    diagnoses_data = {
        'patient_id': [1, 2],
        'admission_id': [300, 400],
        'diagnosis_code': ['D3', 'D4']
    }
    diagnoses_table = pd.DataFrame(diagnoses_data)

    admissions_data = {
        'patient_id': [1,2],
        'admission_id': [300, 400],
        'discharge_time': ['2021-01-05', '2021-02-05']
    }
    admissions_table = pd.DataFrame(admissions_data)
    admissions_table["discharge_time"] = pd.to_datetime(admissions_table["discharge_time"])

    expected_output = {
        1: {
            datetime(2021, 1, 1): 0,
            datetime(2021, 2, 1): 0
        },
        2: {
            datetime(2021, 1, 1): 0
        }
    }

    # Function call
    result = label_lab_records(patient_rolling_lab_records, gap_days, prediction_window_days, positive_diagnoses,
                               diagnoses_table, admissions_table)

    # Assert
    assert result == expected_output


def test_label_lab_records_missing_patients():
    # Sample input data
    patient_rolling_lab_records = {
        1: {
            datetime(2021, 1, 1): pd.DataFrame({"lab_result": [1, 2, 3]}),
            datetime(2021, 2, 1): pd.DataFrame({"lab_result": [4, 5, 6]})
        },
        2: {
            datetime(2021, 1, 1): pd.DataFrame({"lab_result": [7, 8, 9]})
        }
    }

    gap_days = 1
    prediction_window_days = 10
    positive_diagnoses = ['D1', 'D2']

    diagnoses_data = {
        'patient_id': [1],
        'admission_id': [100],
        'diagnosis_code': ['D1']
    }
    diagnoses_table = pd.DataFrame(diagnoses_data)

    admissions_data = {
        'patient_id': [1],
        'admission_id': [100],
        'discharge_time': ['2021-01-05']
    }
    admissions_table = pd.DataFrame(admissions_data)
    admissions_table["discharge_time"] = pd.to_datetime(admissions_table["discharge_time"])

    expected_output = {
        1: {
            datetime(2021, 1, 1): 1,
            datetime(2021, 2, 1): 0
        }
    }

    # Function call
    result = label_lab_records(patient_rolling_lab_records, gap_days, prediction_window_days, positive_diagnoses,
                               diagnoses_table, admissions_table)

    # Assert
    assert result == expected_output


# Save and load functions
@pytest.fixture
def sample_dataframes():
    return {
        'patient1': pd.DataFrame({'name': ['John'], 'age': [30]}),
        'patient2': pd.DataFrame({'name': ['Jane'], 'age': [25]})
    }


def test_save_patient_records_to_pickle(sample_dataframes):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        filename = temp_file.name

    try:
        save_patient_records_to_pickle(sample_dataframes, filename)

        with open(filename, 'rb') as handle:
            loaded_data = pickle.load(handle)

        for key in sample_dataframes:
            assert key in loaded_data
            pd.testing.assert_frame_equal(sample_dataframes[key], loaded_data[key])
    finally:
        os.remove(filename)


def test_load_patient_records_from_pickle(sample_dataframes):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        filename = temp_file.name

    try:
        with open(filename, 'wb') as handle:
            pickle.dump(sample_dataframes, handle)

        loaded_data = load_patient_records_from_pickle(filename)

        for key in sample_dataframes:
            assert key in loaded_data
            pd.testing.assert_frame_equal(sample_dataframes[key], loaded_data[key])
    finally:
        os.remove(filename)


@mock.patch('pandas.DataFrame.to_parquet')
@mock.patch('zipfile.ZipFile')
@mock.patch('os.listdir')
@mock.patch('os.path.join')
def test_save_patient_records_to_parquet_archive(mock_join, mock_listdir, mock_zipfile, mock_to_parquet,
                                                 sample_dataframes):
    archive_name = 'test_data.zip'

    mock_listdir.return_value = ['patient1.parquet', 'patient2.parquet']
    mock_join.side_effect = lambda a, b: f'{a}/{b}'

    with mock_zipfile() as mock_archive:
        save_patient_records_to_parquet_archive(sample_dataframes, archive_name)
        assert mock_zipfile.call_args[0][0] == archive_name
        assert mock_archive.write.call_count == 2
        mock_to_parquet.assert_called()


@mock.patch('pandas.read_parquet')
@mock.patch('zipfile.ZipFile')
@mock.patch('os.listdir')
@mock.patch('os.path.join')
def test_load_patient_records_from_parquet_archive(mock_join, mock_listdir, mock_zipfile, mock_read_parquet):
    archive_name = 'test_data.zip'
    sample_dataframes = {
        'patient1': pd.DataFrame({'name': ['John'], 'age': [30]}),
        'patient2': pd.DataFrame({'name': ['Jane'], 'age': [25]})
    }

    def read_parquet_side_effect(file_path):
        key = os.path.basename(file_path).replace('.parquet', '')
        return sample_dataframes[key]

    mock_listdir.return_value = ['patient1.parquet', 'patient2.parquet']
    mock_join.side_effect = lambda a, b: f'{a}/{b}'
    mock_read_parquet.side_effect = read_parquet_side_effect

    with mock_zipfile() as mock_archive:
        mock_archive.extractall = mock.MagicMock()
        result = load_patient_records_from_parquet_archive(archive_name)
        assert result == sample_dataframes
