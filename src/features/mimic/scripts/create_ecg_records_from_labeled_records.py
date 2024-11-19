import pandas as pd
from tqdm import tqdm

from datasets.mimic_dataset import MimicDataset
from features.mimic.extract_lab_records import load_pickle, save_pickle
from features.mimic.process_records import make_rolling_records
from features.mimic.scripts.data_extraction_config import Config


def create_ecg_features_records_from_labeled_records(extraction_config_file_path, paths_config_filepath,
                                                     dataset_config_file_path):
    # Load config
    with Config(extraction_config_file_path) as cfg:
        with Config(paths_config_filepath) as paths:
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
                save_pickle(rolling_ecg_records, paths.rolling_ecg_records)

    return rolling_ecg_records


def main():
    EXTRACTION_CONFIG_FILEPATH = "config/34-prediction-performance/most_common_analyses/t2d/t2d_B24m_G3m_P5y/t2d_B24m_G3m_P5y_config.json"
    PATH_CONFIG_FILEPATH = "config/34-prediction-performance/most_common_analyses/t2d/t2d_B24m_G3m_P5y/t2d_B24m_G3m_P5y_paths.json"
    MIMIC_DATASET_CONFIG_FILEPATH = "config/mimic_dataset.mipha.json"

    create_ecg_features_records_from_labeled_records(
        extraction_config_file_path=EXTRACTION_CONFIG_FILEPATH,
        paths_config_filepath=PATH_CONFIG_FILEPATH,
        dataset_config_file_path=MIMIC_DATASET_CONFIG_FILEPATH,
    )


if __name__ == '__main__':
    main()
