import os

from tqdm import tqdm

from datasets.mimic_dataset import MimicDataset
from features.mimic.extract_lab_records import load_pickle, save_to_file, find_config_files
from features.mimic.scripts.data_extraction_config import Config


def create_demographics_records_from_labeled_records(extraction_config_file_path, paths_config_filepath,
                                                     dataset_config_file_path, overwrite=False):
    # Load config
    with Config(extraction_config_file_path) as cfg:
        with Config(paths_config_filepath) as paths:
            if os.path.exists(paths.demographics_records):
                if not overwrite:
                    print(f"Existing file found at {paths.demographics_records}. Set overwrite to True to recompute.")
                    return
                else:
                    print(f"Overwriting existing file at {paths.demographics_records}.")

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
                save_to_file(demographics_records, paths.demographics_records)

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
    for patient_id in tqdm(patient_ids, desc="Creating demographic records"):
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
    # Filepaths
    BASE_DIR = "config/34-prediction-performance/categorized_analyses"
    MIMIC_DATASET_CONFIG_FILEPATH = "config/mimic_dataset.mipha.json"

    # Find all config pairs
    # labels will be generated from the first group, so we use the one with the most analyses for safety
    config_pairs = find_config_files(BASE_DIR, first_group="hematology")

    # Display the total number of extractions
    total_pairs = len(config_pairs)
    print(f"Found {total_pairs} config pairs. Starting extraction...\n")

    # Extract data for each config pair
    for index, (config_file, path_config_file) in enumerate(config_pairs, start=1):
        print(f"Running extraction {index}/{total_pairs}")
        create_demographics_records_from_labeled_records(
            extraction_config_file_path=config_file,
            paths_config_filepath=path_config_file,
            dataset_config_file_path=MIMIC_DATASET_CONFIG_FILEPATH,
        )
    print("\nAll extractions completed!")


if __name__ == '__main__':
    main()
