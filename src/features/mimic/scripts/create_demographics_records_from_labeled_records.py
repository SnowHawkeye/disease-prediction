import pandas as pd
from tqdm import tqdm

from datasets.mimic_dataset import MimicDataset
from features.mimic.extract_lab_records import load_pickle, save_pickle
from features.mimic.process_records import make_rolling_records
from features.mimic.scripts.data_extraction_config import Config


def create_demographics_records_from_labeled_records(extraction_config_file_path, paths_config_filepath,
                                                     dataset_config_file_path):
    # Load config
    with Config(extraction_config_file_path) as cfg:
        with Config(paths_config_filepath) as paths:
            print("(1/4) Loading MIMIC dataset...")
            dataset = MimicDataset.from_config_file(dataset_config_file_path)

            print("(2/4) Retrieving observation dates from the provided patient records...")
            patient_records = load_pickle(paths.labeled_lab_records)

            observation_dates = {
                patient_id: [observation_date for observation_date in patient_records[patient_id]]
                for patient_id in patient_records.keys()
            }

            print("(3/4) Loading patients table...")
            patients = dataset.get_patients()
            patients = patients.set_index("patient_id")[["age", "gender", "anchor_year"]]

            print("(4/4) Creating demographic patient records...")
            patient_ids = list(patient_records.keys())

            demographics_records = make_demographic_records(observation_dates, patient_ids, patients)

            if paths.demographics_records is not None:
                print(f"Saving demographics records to {paths.demographics_records}")
                save_pickle(demographics_records, paths.demographics_records)

    return demographics_records


def calculate_age_at_date(patient_demographics, date):
    """
    Given patient information (age, anchor year), returns the age of that patient at a given date.
    """
    birth_date = patient_demographics.anchor_year - patient_demographics.age
    age_at_date = date.year - birth_date
    return age_at_date


def make_demographic_records(observation_dates, patient_ids, patients):
    demographics_records = {}
    for patient_id in patient_ids:
        patient_demographics = patients.loc[patient_id]
        patient_record = {}
        for observation_date in observation_dates[patient_id]:
            patient_record[observation_date] = {
                "age": calculate_age_at_date(patient_demographics, observation_date),
                "gender": patient_demographics.gender
            }
        demographics_records[patient_id] = patient_record
    return demographics_records


def main():
    EXTRACTION_CONFIG_FILEPATH = "config/test_extraction_config.json"
    PATH_CONFIG_FILEPATH = "config/test_extraction_paths.json"
    MIMIC_DATASET_CONFIG_FILEPATH = "config/mimic_dataset.mipha.json"

    create_demographics_records_from_labeled_records(
        extraction_config_file_path=EXTRACTION_CONFIG_FILEPATH,
        paths_config_filepath=PATH_CONFIG_FILEPATH,
        dataset_config_file_path=MIMIC_DATASET_CONFIG_FILEPATH,
    )


if __name__ == '__main__':
    main()
