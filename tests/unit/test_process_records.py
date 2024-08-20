from datetime import datetime

import numpy as np
import pandas as pd

from features.mimic.process_records import make_rolling_records, label_records


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


def test_make_rolling_records_day_set_observation_date():
    # Test with time_unit='day' and backward_window=3
    observations_dates = {
        "patient_1": [
            pd.Timestamp('2023-01-04'),
            pd.Timestamp('2023-01-06'),
        ]
    }

    data = {
        'patient_1': pd.DataFrame({
            'value': [1, 2, 3, 4],
        }, index=pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'])),
    }
    expected = {
        'patient_1': {
            pd.Timestamp('2023-01-04'): pd.DataFrame({'value': [2, 3, 4]},
                                                     index=pd.to_datetime(['2023-01-02', '2023-01-03', '2023-01-04'])),
            pd.Timestamp('2023-01-06'): pd.DataFrame({'value': [4, np.nan, np.nan]},
                                                     index=pd.to_datetime(['2023-01-04', '2023-01-05', '2023-01-06'])),
        },
    }
    result = make_rolling_records(data, 'day', 3, observations_dates)
    for patient_id in result:
        for date in result[patient_id]:
            pd.testing.assert_frame_equal(result[patient_id][date], expected[patient_id][date],
                                          check_dtype=False, check_freq=False)


def test_make_rolling_records_week_set_observation_date():
    # Test with time_unit='week' and backward_window=2
    observations_dates = {
        "patient_1": [
            pd.Timestamp('2023-01-05'),
            pd.Timestamp('2024-05-28'),
        ]
    }

    data = {
        'patient_1': pd.DataFrame({
            'value': [1, 2],
        }, index=pd.to_datetime(['2023-01-08', '2023-01-15']))
    }
    expected = {
        'patient_1': {
            pd.Timestamp('2023-01-05'): pd.DataFrame({'value': [np.nan, 1]},
                                                     index=pd.to_datetime(['2023-01-01', '2023-01-08'])),
            pd.Timestamp('2024-05-28'): pd.DataFrame({'value': [np.nan, np.nan]},
                                                     index=pd.to_datetime(['2024-05-26', '2024-06-02'])),
        }
    }
    result = make_rolling_records(data, 'week', 2, observations_dates)
    for patient_id in result:
        for date in result[patient_id]:
            pd.testing.assert_frame_equal(result[patient_id][date], expected[patient_id][date],
                                          check_dtype=False, check_freq=False)


def test_make_rolling_records_month_set_observation_dates():
    # Test with time_unit='month' and backward_window=2
    observations_dates = {
        "patient_1": [
            pd.Timestamp('2023-01-31'),
            pd.Timestamp('2023-05-28'),
        ]
    }

    data = {
        'patient_1': pd.DataFrame({
            'value': [1, 2, 3, 4],
        }, index=pd.to_datetime(['2023-01-08', '2023-02-14', '2023-04-24', '2023-08-12'])),
        'patient_2': pd.DataFrame({
            'value': [1, 2, 3, 4],
        }, index=pd.to_datetime(['2023-01-08', '2023-02-14', '2023-04-24', '2023-08-12']))
    }
    expected = {
        'patient_1': {
            pd.Timestamp('2023-01-31'): pd.DataFrame({'value': [np.nan, 1]},
                                                     index=pd.to_datetime(['2022-12-31', '2023-01-31'])),
            pd.Timestamp('2023-05-28'): pd.DataFrame({'value': [3, np.nan]},
                                                     index=pd.to_datetime(['2023-04-30', '2023-05-31'])),
        }
    }
    result = make_rolling_records(data, 'month', 2, observation_dates=observations_dates)
    for patient_id in result:
        for date in result[patient_id]:
            pd.testing.assert_frame_equal(result[patient_id][date], expected[patient_id][date],
                                          check_dtype=False, check_freq=False)


def test_make_rolling_records_year_set_observation_dates():
    # Test with time_unit='year' and backward_window=2
    observations_dates = {
        "patient_1": [
            pd.Timestamp('2023-06-06'),
            pd.Timestamp('2024-04-24'),
        ]
    }

    data = {
        'patient_1': pd.DataFrame({
            'value': [1, 2, 3, 4],
        }, index=pd.to_datetime(['2023-01-08', '2024-02-14', '2027-04-24', '2029-08-12']))
    }
    expected = {
        'patient_1': {
            pd.Timestamp('2023-06-06'): pd.DataFrame({'value': [np.nan, 1]},
                                                     index=pd.to_datetime(['2022-12-31', '2023-12-31'])),
            pd.Timestamp('2024-04-24'): pd.DataFrame({'value': [1, 2]},
                                                     index=pd.to_datetime(['2023-12-31', '2024-12-31'])),
        }
    }
    result = make_rolling_records(data, 'year', 2, observations_dates)
    for patient_id in result:
        for date in result[patient_id]:
            pd.testing.assert_frame_equal(result[patient_id][date], expected[patient_id][date],
                                          check_dtype=False, check_freq=False)


def test_label_records():
    # Sample input data
    patient_rolling_records = {
        1: {  # positive patient
            datetime(2020, 1, 1): pd.DataFrame({"lab_result": [1, 2, 3]}),  # too long before the diagnosis
            datetime(2021, 1, 1): pd.DataFrame({"lab_result": [1, 2, 3]}),  # labeled positive
            datetime(2021, 1, 4): pd.DataFrame({"lab_result": [4, 5, 6]}),  # after the first diagnosis, will not appear
            datetime(2021, 3, 1): pd.DataFrame({"lab_result": [4, 5, 6]}),  # after the first diagnosis, will not appear
        },
        2: {  # negative patient
            datetime(2021, 1, 1): pd.DataFrame({"lab_result": [7, 8, 9]}),  # labeled negative
            datetime(2022, 1, 1): pd.DataFrame({"lab_result": [7, 8, 9]})  # labeled negative
        }
    }

    gap_days = 2
    prediction_window_days = 10
    positive_diagnoses = ['D1', 'D2']

    diagnoses_data = {
        'patient_id': [1, 1, 2],
        'admission_id': [100, 101, 200],
        'diagnosis_code': ['D1', 'D3', 'D3']
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
            datetime(2020, 1, 1): 0,
            datetime(2021, 1, 1): 1,
        },
        2: {
            datetime(2021, 1, 1): 0,
            datetime(2022, 1, 1): 0,
        }
    }

    # Function call
    result = label_records(patient_rolling_records, gap_days, prediction_window_days, positive_diagnoses,
                           diagnoses_table, admissions_table)

    # Assert
    assert result == expected_output


def test_label_records_observation_dates_only():
    # Case where there is no data associated to the observation dates

    # Sample input data
    patient_rolling_records = {
        1: [  # positive patient
            datetime(2020, 1, 1),  # too long before the diagnosis
            datetime(2021, 1, 1),  # labeled positive
            datetime(2021, 1, 4),  # after the first diagnosis, will not appear
            datetime(2021, 3, 1),  # after the first diagnosis, will not appear
        ]

        ,
        2: [  # negative patient
            datetime(2021, 1, 1),  # labeled negative
            datetime(2022, 1, 1),  # labeled negative
        ]

    }

    gap_days = 2
    prediction_window_days = 10
    positive_diagnoses = ['D1', 'D2']

    diagnoses_data = {
        'patient_id': [1, 1, 2],
        'admission_id': [100, 101, 200],
        'diagnosis_code': ['D1', 'D3', 'D3']
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
            datetime(2020, 1, 1): 0,
            datetime(2021, 1, 1): 1,
        },
        2: {
            datetime(2021, 1, 1): 0,
            datetime(2022, 1, 1): 0,
        }
    }

    # Function call
    result = label_records(patient_rolling_records, gap_days, prediction_window_days, positive_diagnoses,
                           diagnoses_table, admissions_table)

    # Assert
    assert result == expected_output


def test_label_records_no_diagnoses():
    # Sample input data
    patient_rolling_records = {
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
        'patient_id': [1, 2],
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
    result = label_records(patient_rolling_records, gap_days, prediction_window_days, positive_diagnoses,
                           diagnoses_table, admissions_table)

    # Assert
    assert result == expected_output


def test_label_records_missing_patients():
    # Sample input data
    patient_rolling_records = {
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
        }
    }

    # Function call
    result = label_records(patient_rolling_records, gap_days, prediction_window_days, positive_diagnoses,
                           diagnoses_table, admissions_table)

    # Assert
    assert result == expected_output
