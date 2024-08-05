import pandas as pd

from datasets.mimic_dataset import MimicDataset
from features.mimic.extract_lab_records import extract_lab_records, save_patient_records_to_pickle


def get_sample_analyses():
    return pd.read_parquet("data/analyses_sample.parquet")


def main():
    dataset = MimicDataset.from_config_file("config/mimic_dataset.mipha.json")
    columns = dataset.column_aliases

    print("Loading analyses table...")
    dataset.get_analyses = get_sample_analyses
    analyses = dataset.get_analyses()
    print("Analyses table loaded!")

    patient_dataframes = extract_lab_records(analyses=analyses, cols=columns)

    save_patient_records_to_pickle(patient_dataframes, "data/sample_patient_lab_records.pkl")


if __name__ == '__main__':
    main()
