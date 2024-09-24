import json

extraction_config = {
    "selected_analyses_ids": None,
    "positive_diagnoses": [],  # list of icd codes given as strings
    "lab_records_frequency": "h",  # cf. https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html
    "backward_window_time_unit": "month",  # possible values are: {'day', 'week', 'month', 'year'}
    "backward_window_value": 1,  # given in the unit of backward_window_time_unit
    "gap_days": 31,  # the value must be given in days
    "prediction_window_days": 31,  # the value must be given in days
}

paths_config = {
    "lab_records": None,
    "filtered_analyses": None,
    "filtered_lab_records": None,
    "rolling_lab_records": None,
    "labeled_lab_records": None,
    "rolling_ecg_records": None,
    "demographics_records": None,
}


def generate_config(config: dict, filepath):
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=4)


class Config:
    def __init__(self, filepath):
        with open(filepath, 'r') as f:
            self._data = json.load(f)

    def __getattr__(self, item):
        if item in self._data:
            return self._data[item]
        raise AttributeError(f"Config object has no attribute '{item}'")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def get(self, key):
        return self._data[key]


def main():
    EXTRACTION_CONFIG_FILEPATH = "config/34-prediction-performance/t2d/t2d_B24m_G3m_P5y/t2d_B24m_G3m_P5y_config.json"
    PATH_CONFIG_FILEPATH = "config/34-prediction-performance/t2d/t2d_B24m_G3m_P5y/t2d_B24m_G3m_P5y_paths.json"

    generate_config(extraction_config, EXTRACTION_CONFIG_FILEPATH)
    generate_config(paths_config, PATH_CONFIG_FILEPATH)


if __name__ == '__main__':
    main()
