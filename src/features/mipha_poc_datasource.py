import json
import pickle
from collections import namedtuple
from os import path

import pandas as pd
from mipha.framework import DataSource
from sklearn.model_selection import train_test_split

from src.datasets.mimic_dataset import MimicDataset


class Stage5CkdData:
    def __init__(self, dataset: MimicDataset, processed_data_root):
        """
        :param dataset: The dataset to fetch data from.
        """
        self.dataset = dataset
        self.processed_data_root = processed_data_root

    def _split_data(self, x, y, meta, test_size=0.2, random_state=None):
        """
        Split data between train and test set based on the patients' IDs.
        :return: x_train, x_test, y_train, y_test, meta_train, meta_test
        """
        df_id = pd.DataFrame([t["patient_id"] for t in meta])

        train, test = train_test_split(df_id.drop_duplicates(), test_size=test_size, random_state=random_state)
        indices_train = pd.merge(df_id.reset_index(), train, on=0)["index"]
        indices_test = pd.merge(df_id.reset_index(), test, on=0)["index"]

        x_train = [x[i] for i in indices_train]
        x_test = [x[i] for i in indices_test]

        y_train = [y[i] for i in indices_train]
        y_test = [y[i] for i in indices_test]

        meta_train = [meta[i] for i in indices_train]
        meta_test = [meta[i] for i in indices_test]

        return x_train, x_test, y_train, y_test, meta_train, meta_test

    def _get_demographics(self, analysis_metadata, patients_table):
        """Given the metadata of an analysis (subject_id and storetime),
        computes the age of the corresponding patient at the time of the analysis.
        Does not account for exact birthdate, but a +/-1 year difference is accepted"""
        # FIXME we assume the id is effectively in the patients table

        row = patients_table[patients_table["patient_id"] == analysis_metadata["patient_id"]].iloc[0][
            ["gender", "anchor_age", "anchor_year"]]
        current_age = (
                row["anchor_age"] +  # age used as reference
                pd.to_datetime(analysis_metadata["analysis_time"]).year -  # year when the analysis was performed
                pd.to_datetime(row["anchor_year"], format="%Y").year  # year used as reference
        )
        return {"gender": row["gender"], "age": current_age}

    def _load_processed_data(self, processed_data_name, random_state):
        """
        :return: Biological data retrieved from the given dataset, split between train and test set.
        """
        # Load the relevant tables
        patients = self.dataset.get_table(module="hospital", table="patients")
        positive_patients_path = path.join(self.processed_data_root, processed_data_name,
                                           'positive_patients_matrices.pkl')
        negative_patients_path = path.join(self.processed_data_root, processed_data_name,
                                           'negative_patients_matrices.pkl')

        with open(positive_patients_path, 'rb') as handle:
            x_pos, y_pos, meta_pos = pickle.load(handle)
        with open(negative_patients_path, 'rb') as handle:
            x_neg, y_neg, meta_neg = pickle.load(handle)

        # Split between training and test set
        x_train_pos, x_test_pos, y_train_pos, y_test_pos, meta_train_pos, meta_test_pos \
            = self._split_data(x_pos, y_pos, meta_pos, random_state=random_state)
        x_train_neg, x_test_neg, y_train_neg, y_test_neg, meta_train_neg, meta_test_neg \
            = self._split_data(x_neg, y_neg, meta_neg, random_state=random_state)

        # Rejoin positive and negative samples
        bio_data_train, labels_train, meta_train \
            = x_train_pos + x_train_neg, y_train_pos + y_train_neg, meta_train_pos + meta_train_neg
        bio_data_test, labels_test, meta_test \
            = x_test_pos + x_test_neg, y_test_pos + y_test_neg, meta_test_pos + meta_test_neg

        return bio_data_test, bio_data_train, labels_test, labels_train, meta_test, meta_train, patients

    @staticmethod
    def create_config_file(config_path):
        """
        Creates a configuration file for the dataset provider.

        Parameters:
        - config_path (str): Path to save the configuration file.
        """
        config = {
            'processed_data_root': ''
        }

        with open(config_path, 'w') as config_file:
            json.dump(config, config_file, indent=4, sort_keys=True)
        print(f"Default configuration file for MIPHA POC data (stage 5 CKD) created at {config_path}.")
        print(f"Make sure to change the path to point at the processed data root.")

    @staticmethod
    def from_config_file(dataset: MimicDataset, config_path):
        """
        Loads configuration from a file and initializes StageFiveCkdData.
        :param dataset: MimicDataset to call for data fetching.
        :param config_path: Path to the configuration file.
        :returns: `StageFiveCkdData` instance.
        """
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)

        processed_data_root = config.get("processed_data_root")
        return Stage5CkdData(dataset=dataset, processed_data_root=processed_data_root)

    def load_stage_5_ckd(self, random_state=None):
        bio_data_test, bio_data_train, labels_test, labels_train, meta_test, meta_train, patients \
            = self._load_processed_data(processed_data_name="stage_5_ckd", random_state=random_state)

        # Create data sources
        data_source_1_train = DataSource(
            data_type="Creatinine",
            data=[pd.DataFrame(t["analysis_50912"]) for t in bio_data_train],
            name="source1_train",
        )

        data_source_2_train = DataSource(
            data_type="Demographics",
            data=pd.DataFrame([self._get_demographics(t, patients) for t in meta_train]),
            name="source2_train",
        )

        data_source_1_test = DataSource(
            data_type="Creatinine",
            data=[pd.DataFrame(t["analysis_50912"]) for t in bio_data_test],
            name="source1_test",
        )

        data_source_2_test = DataSource(
            data_type="Demographics",
            data=pd.DataFrame([self._get_demographics(t, patients) for t in meta_test]),
            name="source2_test",
        )

        LearningData = namedtuple(
            typename="LearningData",
            field_names=["data_sources_train", "labels_train", "data_sources_test", "labels_test"]
        )

        return LearningData(
            data_sources_train=[data_source_1_train, data_source_2_train],
            labels_train=labels_train,
            data_sources_test=[data_source_1_test, data_source_2_test],
            labels_test=labels_test,
        )
