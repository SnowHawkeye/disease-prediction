import pandas as pd
from datetime import datetime

from features.mimic.scripts.create_demographics_records_from_labeled_records import make_demographic_records


def test_make_demographic_records():
    # Sample data
    patients = pd.DataFrame({
        'gender': ['F', 'M'],
        'age': [52, 60],
        'anchor_year': [2180, 2180]
    }, index=[1, 2])

    observation_dates = {
        1: [datetime(2181, 1, 1), datetime(2182, 1, 1)],
        2: [datetime(2181, 1, 1)]
    }

    patient_ids = [1, 2]

    # Call the function
    result = make_demographic_records(observation_dates, patient_ids, patients)

    # Expected output
    expected = {
        1: {
            datetime(2181, 1, 1): {"age": 53, "gender": 'F'},
            datetime(2182, 1, 1): {"age": 54, "gender": 'F'}
        },
        2: {
            datetime(2181, 1, 1): {"age": 61, "gender": 'M'}
        }
    }

    assert result == expected
