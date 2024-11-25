import os
from os import path

from datasets.mimic_dataset import MimicDataset
from features.mimic.extract_lab_records import extract_lab_records, load_pickle, filter_lab_records, save_to_file, \
    find_config_files
from features.mimic.process_records import make_rolling_records, label_records
from features.mimic.scripts.data_extraction_config import Config


def extract_data(extraction_config_file_path, paths_config_filepath, dataset, analyses_table):
    """
    Extract data according to parameters given in the config files.
    The script automatically advances to the furthest step for which a result has been provided
    (i.e. a file already exists at the path where the step's result has to be saved).
    Make sure no file exists for the last step (labeled_lab_records).
    :param extraction_config_file_path: Config file for extraction parameters
    :param paths_config_filepath: Config file for save and load paths
    :param dataset: Dataset to extract the data from
    :param analyses_table: Pre-loaded table containing the analyses
    """

    # Load config
    with Config(extraction_config_file_path) as cfg:
        with Config(paths_config_filepath) as paths:

            # Load dataset
            print("(1/5) Initializing extraction...")

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

                # Labeling is shared by several extractions, so steps shouldn't be skipped based on that
                for future_step, _ in steps[i - 1:-1]:
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
                        results[step] = make_func(cfg, dataset, paths, analyses_table)
                    elif step == "filtered_lab_records":
                        results[step] = make_func(cfg, results["lab_records"], paths)
                    elif step == "rolling_lab_records":
                        results[step] = make_func(cfg, results["filtered_lab_records"], paths)
                    elif step == "labeled_lab_records":
                        results[step] = make_func(cfg, dataset, results["rolling_lab_records"], paths)


def make_lab_records(cfg, dataset, paths, analyses_table=None):
    def load_analyses_table():
        if analyses_table is None:
            print("Loading analyses table...")
            loaded_analyses = dataset.get_analyses()
            print("Analyses table loaded!")
        else:
            print("Loading the provided analyses table...")
            loaded_analyses = analyses_table
        return loaded_analyses

    if cfg.selected_analyses_ids is not None:  # filter analyses for selection
        if paths.filtered_analyses is not None and path.exists(paths.filtered_analyses):
            print(f"Loading filtered analyses table from {paths.filtered_analyses}")
            analyses = load_pickle(paths.filtered_analyses)
        else:
            analyses = load_analyses_table()
            print("Filtering analyses table...")
            analyses = analyses[analyses["analysis_id"].isin(cfg.selected_analyses_ids)]
            if paths.filtered_analyses is not None:
                print(f"Saving filtered analyses table to {paths.filtered_analyses}")
                save_to_file(analyses, paths.filtered_analyses)
    else:  # no analyses provided for filtering
        analyses = load_analyses_table()

    lab_records = extract_lab_records(
        analyses=analyses,
        freq=cfg.lab_records_frequency,
        aggfunc='mean'
    )
    if paths.lab_records is not None:
        print(f"Saving lab records to {paths.lab_records}")
        save_to_file(lab_records, paths.lab_records)
    return lab_records


def make_filtered_lab_records(cfg, lab_records, paths):
    filtered_lab_records = filter_lab_records(
        patient_lab_records=lab_records,
        analyses_ids=cfg.selected_analyses_ids,
    )
    if paths.filtered_lab_records is not None:
        print(f"Saving filtered lab records to {paths.filtered_lab_records}")
        save_to_file(filtered_lab_records, paths.filtered_lab_records)
    return filtered_lab_records


def make_rolling_lab_records(cfg, filtered_lab_records, paths):
    rolling_lab_records = make_rolling_records(
        patient_records=filtered_lab_records,
        time_unit=cfg.backward_window_time_unit,
        backward_window=cfg.backward_window_value
    )
    if paths.rolling_lab_records is not None:
        print(f"Saving rolling lab records to {paths.rolling_lab_records}")
        save_to_file(rolling_lab_records, paths.rolling_lab_records)
    return rolling_lab_records


def label_rolling_lab_records(cfg, dataset, rolling_lab_records, paths):
    diagnoses_table = dataset.get_diagnoses()
    admissions_table = dataset.get_admissions()
    labeled_lab_records = label_records(
        patient_rolling_records=rolling_lab_records,
        gap_days=cfg.gap_days,
        prediction_window_days=cfg.prediction_window_days,
        positive_diagnoses=cfg.positive_diagnoses,
        diagnoses_table=diagnoses_table,
        admissions_table=admissions_table,
    )
    if paths.labeled_lab_records is not None:
        print(f"Saving labeled lab records to {paths.labeled_lab_records}")
        save_to_file(labeled_lab_records, paths.labeled_lab_records)
    return labeled_lab_records


def main():
    # Filepaths
    BASE_DIR = "config/34-prediction-performance/categorized_analyses"
    MIMIC_DATASET_CONFIG_FILEPATH = "config/mimic_dataset.mipha.json"

    dataset = MimicDataset.from_config_file(MIMIC_DATASET_CONFIG_FILEPATH)
    print("Pre-loading the analyses table...")
    analyses_table = dataset.get_analyses()

    # Find all config pairs
    # labels will be generated from the first group, so we use the one with the most analyses for safety
    config_pairs = find_config_files(BASE_DIR, first_group="hematology")

    # Display the total number of extractions
    total_pairs = len(config_pairs)
    print(f"Found {total_pairs} config pairs. Starting extraction...\n")

    # Extract data for each config pair
    for index, (config_file, path_config_file) in enumerate(config_pairs, start=1):
        print(f"Running extraction {index}/{total_pairs}")
        extract_data(
            extraction_config_file_path=config_file,
            paths_config_filepath=path_config_file,
            dataset=dataset,
            analyses_table=analyses_table,
        )
    print("\nAll extractions completed!")


if __name__ == "__main__":
    main()
