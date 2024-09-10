from datasets.mimic_dataset import MimicDataset
from features.mimic.extract_lab_records import extract_lab_records, save_pickle


def main():
    dataset = MimicDataset.from_config_file("../config/mimic_dataset.mipha.json")

    print("Loading analyses table...")
    analyses = dataset.get_analyses()
    print("Analyses table loaded!")

    patient_dataframes = extract_lab_records(
        analyses=analyses,
        freq='h',
        aggfunc='mean'
    )

    save_pickle(patient_dataframes, "data/patient_lab_records.pkl")


if __name__ == '__main__':
    main()
