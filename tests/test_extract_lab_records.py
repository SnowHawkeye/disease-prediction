import os
import pickle
import tempfile
from unittest import mock

import pandas as pd
import pytest

from src.features.mimic.extract_lab_records import extract_lab_records, save_patient_records_to_parquet_archive, \
    load_pickle, \
    save_pickle, load_patient_records_from_parquet_archive, filter_lab_records


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
        save_pickle(sample_dataframes, filename)

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

        loaded_data = load_pickle(filename)

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
