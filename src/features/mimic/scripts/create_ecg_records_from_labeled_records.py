import os

import pandas as pd
from tqdm import tqdm

from datasets.mimic_dataset import MimicDataset
from features.mimic.extract_lab_records import load_pickle, save_to_file, find_config_files
from features.mimic.process_records import make_rolling_records
from features.mimic.scripts.data_extraction_config import Config


def create_ecg_features_records_from_labeled_records(extraction_config_file_path, paths_config_filepath,
                                                     dataset_config_file_path, overwrite=False):
    # Load config
    with Config(extraction_config_file_path) as cfg:
        with Config(paths_config_filepath) as paths:
            if os.path.exists(paths.rolling_ecg_records):
                if not overwrite:
                    print(f"Existing file found at {paths.rolling_ecg_records}. Set overwrite to True to recompute.")
                    return
                else:
                    print(f"Overwriting existing file at {paths.rolling_ecg_records}.")

            print("(1/5) Loading MIMIC dataset...")
            dataset = MimicDataset.from_config_file(dataset_config_file_path)

            print("(2/5) Retrieving observation dates from the provided patient records...")
            patient_records = load_pickle(paths.labeled_lab_records)

            observation_dates = {
                patient_id: [observation_date for observation_date in patient_records[patient_id]]
                for patient_id in patient_records.keys()
            }

            print("(3/5) Loading ECG machine measurements table...")
            machine_measurements = dataset.get_ecg_machine_measurements()
            machine_measurements['ecg_time'] = pd.to_datetime(machine_measurements['ecg_time'])

            kept_features = [
                'rr_interval', 'p_onset', 'p_end', 'qrs_onset', 'qrs_end', 't_end', 'p_axis', 'qrs_axis', 't_axis'
            ]

            ecg_features = machine_measurements.set_index("ecg_time")[['patient_id'] + kept_features]

            print("(4/5) Creating ECG measurements patient records...")
            patient_ids = list(patient_records.keys())
            ecg_features = ecg_features[ecg_features['patient_id'].isin(patient_ids)]

            grouped_data = {
                patient_id: data.drop(columns='patient_id')
                for patient_id, data in tqdm(ecg_features.groupby('patient_id'), desc="Creating ECG records")
            }

            print("(5/5) Making rolling records from ECG machine measurements...")
            rolling_ecg_records = make_rolling_records(
                patient_records=grouped_data,
                time_unit=cfg.backward_window_time_unit,
                backward_window=cfg.backward_window_value,
                observation_dates=observation_dates
            )

            if paths.rolling_ecg_records is not None:
                print(f"Saving rolling ecg records to {paths.rolling_ecg_records}")
                save_to_file(rolling_ecg_records, paths.rolling_ecg_records)


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
        create_ecg_features_records_from_labeled_records(
            extraction_config_file_path=config_file,
            paths_config_filepath=path_config_file,
            dataset_config_file_path=MIMIC_DATASET_CONFIG_FILEPATH,
        )
    print("\nAll extractions completed!")


if __name__ == '__main__':
    main()
