import os
from os import path

from datasets.mimic_dataset import MimicDataset
from features.mimic.extract_lab_records import extract_lab_records, load_pickle, save_pickle, filter_lab_records, \
    make_rolling_records, label_lab_records
from features.mimic.scripts.labeled_lab_data_extraction_config import Config


def extract_data(extraction_config_file_path, paths_config_filepath, dataset_config_file_path):
    """
    Extract data according to parameters given in the config files.
    The script automatically advances to the furthest step for which a result has been provided
    (i.e. a file already exists at the path where the step's result has to be saved)
    :param extraction_config_file_path: Config file for extraction parameters
    :param paths_config_filepath: Config file for save and load paths
    :param dataset_config_file_path: Config file for dataset
    :return:
    """

    # Load config
    with Config(extraction_config_file_path) as cfg:
        with Config(paths_config_filepath) as paths:

            # Load dataset
            print("(1/5) Loading MIMIC dataset...")
            dataset = MimicDataset.from_config_file(dataset_config_file_path)

            steps = [
                ("lab_records", make_lab_records),
                ("filtered_lab_records", make_filtered_lab_records),
                ("rolling_lab_records", make_rolling_lab_records),
                ("labeled_lab_records", label_rolling_lab_records),
            ]

            results = {}
            for i, (step, make_func) in enumerate(steps, start=2):
                # Check if any of the following steps are done
                skip_current_step = False
                for future_step, _ in steps[i - 1:]:
                    future_file_path = paths.get(future_step)
                    if os.path.exists(future_file_path):
                        print(f"({i}/5) Skipping step {i} ({step}), because subsequent steps are already done.")
                        skip_current_step = True
                        break

                if skip_current_step:
                    continue

                file_path = paths.get(step)
                if os.path.exists(file_path):
                    print(f"({i}/5) Skipping step {i} ({step}) as it was already done.")
                    results[step] = load_pickle(file_path)
                else:
                    print(f"({i}/5) Executing step {i} ({step})...")
                    if step == "lab_records":
                        results[step] = make_func(cfg, dataset, paths)
                    elif step == "filtered_lab_records":
                        results[step] = make_func(cfg, results["lab_records"], paths)
                    elif step == "rolling_lab_records":
                        results[step] = make_func(cfg, results["filtered_lab_records"], paths)
                    elif step == "labeled_lab_records":
                        results[step] = make_func(cfg, dataset, results["rolling_lab_records"], paths)

            return results["rolling_lab_records"], results["labeled_lab_records"]


def make_lab_records(cfg, dataset, paths):
    if cfg.selected_analyses_ids is not None:  # filter analyses for selection
        if paths.filtered_analyses is not None and path.exists(paths.filtered_analyses):
            print(f"Loading filtered analyses table from {paths.filtered_analyses}")
            analyses = load_pickle(paths.filtered_analyses)
        else:
            print("Loading analyses table...")
            analyses = dataset.get_analyses()
            print("Analyses table loaded!")

            print("Filtering analyses table...")
            analyses = analyses[analyses["analysis_id"].isin(cfg.selected_analyses_ids)]
            if paths.filtered_analyses is not None:
                print(f"Saving filtered analyses table to {paths.filtered_analyses}")
                save_pickle(analyses, paths.filtered_analyses)
    else:
        print("Loading analyses table...")
        analyses = dataset.get_analyses()
        print("Analyses table loaded!")

    lab_records = extract_lab_records(
        analyses=analyses,
        freq=cfg.lab_records_frequency,
        aggfunc='mean'
    )
    if paths.lab_records is not None:
        print(f"Saving lab records to {paths.lab_records}")
        save_pickle(lab_records, paths.lab_records)
    return lab_records


def make_filtered_lab_records(cfg, lab_records, paths):
    filtered_lab_records = filter_lab_records(
        patient_lab_records=lab_records,
        analyses_ids=cfg.selected_analyses_ids,
    )
    if paths.filtered_lab_records is not None:
        print(f"Saving filtered lab records to {paths.filtered_lab_records}")
        save_pickle(filtered_lab_records, paths.filtered_lab_records)
    return filtered_lab_records


def make_rolling_lab_records(cfg, filtered_lab_records, paths):
    rolling_lab_records = make_rolling_records(
        patient_lab_records=filtered_lab_records,
        time_unit=cfg.backward_window_time_unit,
        backward_window=cfg.backward_window_value
    )
    if paths.rolling_lab_records is not None:
        print(f"Saving rolling lab records to {paths.rolling_lab_records}")
        save_pickle(rolling_lab_records, paths.rolling_lab_records)
    return rolling_lab_records


def label_rolling_lab_records(cfg, dataset, rolling_lab_records, paths):
    diagnoses_table = dataset.get_diagnoses()
    admissions_table = dataset.get_admissions()
    labeled_lab_records = label_lab_records(
        patient_rolling_lab_records=rolling_lab_records,
        gap_days=cfg.gap_days,
        prediction_window_days=cfg.prediction_window_days,
        positive_diagnoses=cfg.positive_diagnoses,
        diagnoses_table=diagnoses_table,
        admissions_table=admissions_table,
    )
    if paths.labeled_lab_records is not None:
        print(f"Saving labeled lab records to {paths.labeled_lab_records}")
        save_pickle(labeled_lab_records, paths.labeled_lab_records)
    return labeled_lab_records


def main():
    EXTRACTION_CONFIG_FILEPATH = "config/test_extraction_config.json"
    PATH_CONFIG_FILEPATH = "config/test_extraction_paths.json"
    MIMIC_DATASET_CONFIG_FILEPATH = "config/mimic_dataset.mipha.json"

    extract_data(
        extraction_config_file_path=EXTRACTION_CONFIG_FILEPATH,
        paths_config_filepath=PATH_CONFIG_FILEPATH,
        dataset_config_file_path=MIMIC_DATASET_CONFIG_FILEPATH,
    )


if __name__ == '__main__':
    main()
