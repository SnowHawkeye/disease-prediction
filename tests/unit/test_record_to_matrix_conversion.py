import numpy as np
import pandas as pd
import pytest

from features.mimic.record_to_matrix_conversion import create_label_matrix, create_mask_from_records, \
    create_learning_matrix

# Convention: timestamp[patient_number][timestamp_number]
timestamp11 = pd.Timestamp("2016-02-29 13:00")
timestamp21 = pd.Timestamp("2025-01-29 12:00")
timestamp22 = pd.Timestamp("2028-05-12 19:30")
timestamp41 = pd.Timestamp("2020-06-04 02:00")
timestamp42 = pd.Timestamp("2020-06-04 08:00")
timestamp43 = pd.Timestamp("2023-11-11 11:11")


def make_record(timestamp, values):
    """
    Given a mock observation date and values, returns a dataframe with 3 timestamps and the same values.
    Serves as mock record (backward window of 3 months, 2 features)
    """
    timestamp3 = timestamp
    timestamp2 = timestamp3 - pd.DateOffset(months=1)
    timestamp1 = timestamp2 - pd.DateOffset(months=1)

    return pd.DataFrame.from_dict({
        timestamp1: values,
        timestamp2: values,
        timestamp3: values,
    }, orient='index')


@pytest.fixture
def mock_rolling_records():
    data = {
        "patient_id_1": {
            timestamp11: make_record(timestamp11, {"f1": 1.11, "f2": 1.12}),
        },
        "patient_id_2": {
            timestamp21: make_record(timestamp21, {"f1": 2.11, "f2": 2.12}),
            timestamp22: make_record(timestamp22, {"f1": 2.21, "f2": 2.22}),
        },
        "patient_id_3": {},
        "patient_id_4": {
            timestamp41: make_record(timestamp41, {"f1": np.NaN, "f2": np.NaN}),
            timestamp42: make_record(timestamp42, {"f1": 4.21, "f2": np.NaN}),
            timestamp43: make_record(timestamp43, {"f1": 4.31, "f2": 4.32}),
        },
    }
    return data


@pytest.fixture
def mock_labels():
    data = {
        "patient_id_1": {
            timestamp11: 0,
        },
        "patient_id_2": {
            timestamp21: 1,
            timestamp22: 1,
        },
        "patient_id_3": {},
        "patient_id_4": {
            timestamp41: 0,
            timestamp42: 0,
            timestamp43: 1,
        },
    }
    return data


def test_create_mask_from_labels(mock_labels):
    # GIVEN
    expected = [
        ("patient_id_1", timestamp11),
        ("patient_id_2", timestamp21),
        ("patient_id_2", timestamp22),
        ("patient_id_4", timestamp41),
        ("patient_id_4", timestamp42),
        ("patient_id_4", timestamp43),
    ]

    # WHEN
    result = create_mask_from_records(mock_labels)

    # THEN
    assert result == expected


def test_create_mask_from_records(mock_rolling_records):
    # GIVEN
    expected = [
        ("patient_id_1", timestamp11),
        ("patient_id_2", timestamp21),
        ("patient_id_2", timestamp22),
        ("patient_id_4", timestamp41),
        ("patient_id_4", timestamp42),
        ("patient_id_4", timestamp43),
    ]

    # WHEN
    result = create_mask_from_records(mock_rolling_records)

    # THEN
    assert result == expected


def test_create_label_matrix(mock_labels):
    # GIVEN
    mask = [
        ("patient_id_1", timestamp11),
        ("patient_id_2", timestamp21),
        ("patient_id_2", timestamp22),
        ("patient_id_4", timestamp41),
        ("patient_id_4", timestamp42),
        ("patient_id_4", timestamp43),
    ]
    expected = np.array([0, 1, 1, 0, 0, 1])

    # WHEN
    result = create_label_matrix(mock_labels, mask)

    # THEN
    np.testing.assert_array_equal(result, expected)


def test_create_learning_matrix(mock_rolling_records):
    # GIVEN
    mask = [
        ("patient_id_1", timestamp11),
        ("patient_id_2", timestamp21),
        ("patient_id_2", timestamp22),
        ("patient_id_4", timestamp41),
        ("patient_id_4", timestamp42),
        ("patient_id_4", timestamp43),
    ]
    expected = np.array(  # 3D array of shape (len_mask, backward_window, features)
        [
            [  # ("patient_id_1", timestamp11)
                [1.11, 1.12],
                [1.11, 1.12],
                [1.11, 1.12],
            ],
            [  # ("patient_id_2", timestamp21)
                [2.11, 2.12],
                [2.11, 2.12],
                [2.11, 2.12],
            ],
            [  # ("patient_id_2", timestamp22)
                [2.21, 2.22],
                [2.21, 2.22],
                [2.21, 2.22],
            ],
            [  # ("patient_id_4", timestamp41)
                [np.NaN, np.NaN],
                [np.NaN, np.NaN],
                [np.NaN, np.NaN],
            ],
            [  # ("patient_id_4", timestamp42)
                [4.21, np.NaN],
                [4.21, np.NaN],
                [4.21, np.NaN],
            ],
            [  # ("patient_id_4", timestamp43)
                [4.31, 4.32],
                [4.31, 4.32],
                [4.31, 4.32],
            ],
        ]

    )

    # WHEN
    result = create_learning_matrix(mock_rolling_records, mask)

    # THEN
    np.testing.assert_array_equal(result, expected)


def test_create_learning_matrix_missing_patient_id(mock_rolling_records):
    # GIVEN
    mask = [
        ("patient_id_1", timestamp11),
        ("patient_id_2", timestamp21),
        ("patient_id_2", timestamp22),
        ("patient_id_4", timestamp41),
        ("patient_id_4", timestamp42),
        ("patient_id_4", timestamp43),
        ("extra_patient", pd.Timestamp("2020-01-01")),
    ]
    expected = np.array(  # 3D array of shape (len_mask, backward_window, features)
        [
            [  # ("patient_id_1", timestamp11)
                [1.11, 1.12],
                [1.11, 1.12],
                [1.11, 1.12],
            ],
            [  # ("patient_id_2", timestamp21)
                [2.11, 2.12],
                [2.11, 2.12],
                [2.11, 2.12],
            ],
            [  # ("patient_id_2", timestamp22)
                [2.21, 2.22],
                [2.21, 2.22],
                [2.21, 2.22],
            ],
            [  # ("patient_id_4", timestamp41)
                [np.NaN, np.NaN],
                [np.NaN, np.NaN],
                [np.NaN, np.NaN],
            ],
            [  # ("patient_id_4", timestamp42)
                [4.21, np.NaN],
                [4.21, np.NaN],
                [4.21, np.NaN],
            ],
            [  # ("patient_id_4", timestamp43)
                [4.31, 4.32],
                [4.31, 4.32],
                [4.31, 4.32],
            ],
            [  # ("extra_patient", pd.Timestamp("2020-01-01"))
                [np.NaN, np.NaN],
                [np.NaN, np.NaN],
                [np.NaN, np.NaN],
            ],
        ]

    )

    # WHEN
    result = create_learning_matrix(mock_rolling_records, mask)

    # THEN
    np.testing.assert_array_equal(result, expected)


def test_create_learning_matrix_missing_timestamp(mock_rolling_records):
    # GIVEN
    mask = [
        ("patient_id_1", timestamp11),
        ("patient_id_2", timestamp21),
        ("patient_id_2", timestamp22),
        ("patient_id_4", timestamp41),
        ("patient_id_4", timestamp42),
        ("patient_id_4", timestamp43),
        ("patient_id_4", pd.Timestamp("2020-01-01")), # this timestamp is not in records
    ]
    expected = np.array(  # 3D array of shape (len_mask, backward_window, features)
        [
            [  # ("patient_id_1", timestamp11)
                [1.11, 1.12],
                [1.11, 1.12],
                [1.11, 1.12],
            ],
            [  # ("patient_id_2", timestamp21)
                [2.11, 2.12],
                [2.11, 2.12],
                [2.11, 2.12],
            ],
            [  # ("patient_id_2", timestamp22)
                [2.21, 2.22],
                [2.21, 2.22],
                [2.21, 2.22],
            ],
            [  # ("patient_id_4", timestamp41)
                [np.NaN, np.NaN],
                [np.NaN, np.NaN],
                [np.NaN, np.NaN],
            ],
            [  # ("patient_id_4", timestamp42)
                [4.21, np.NaN],
                [4.21, np.NaN],
                [4.21, np.NaN],
            ],
            [  # ("patient_id_4", timestamp43)
                [4.31, 4.32],
                [4.31, 4.32],
                [4.31, 4.32],
            ],
            [  # ("patient_id_4", pd.Timestamp("2020-01-01"))
                [1, 1],
                [1, 1],
                [1, 1],
            ],
        ]

    )

    # WHEN
    result = create_learning_matrix(mock_rolling_records, mask, fill_value=1)

    # THEN
    np.testing.assert_array_equal(result, expected)