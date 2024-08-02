import json
from os import path

import pandas as pd


class MimicDataset:
    """
    Data manager for the MIMIC-IV database. The default names for tables and columns are sourced from version 2.2 of the database.
    """

    def __init__(self, config):
        """
        Initializes the MimicDataManager with the provided configuration.
        :param config: A dictionary containing the configuration settings.
        """

        version = "2.2"

        # keys are the "normalized" names, values are the corresponding names in the database
        # utility to find columns more easily
        column_aliases = {
            'diagnosis_code': 'icd_code',
            'admission_id': 'hadm_id',
            'patient_id': 'subject_id',
            'analysis_id': 'itemid',
            'specimen_id': 'specimen_id',
            'admission_time': 'admittime',
            'age': 'anchor_age',
            'diagnosis_time': 'diagnosis_time',
            'analysis_time': 'storetime',
            'analysis_name': 'label',
            'analysis_value': 'valuenum',
        }

        tables = {
            "analyses_lookup": "d_labitems.csv.gz",
            "diagnoses_lookup": "d_icd_diagnoses.csv.gz",
            "patients": "patients.csv.gz",
            "diagnoses": "diagnoses_icd.csv.gz",
            "analyses": "labevents.csv.gz"
        }

        modules = {
            "hospital": "hosp",
        }

        load_params = {
            'delimiter': ',',
            'encoding': 'utf-8',
        }

        self.config = config
        self.database_path = config.get('database_root', '')
        self.delimiter = load_params.get('delimiter', ',')
        self.encoding = load_params.get('encoding', 'utf-8')
        self.column_aliases = column_aliases
        self.tables = tables
        self.version = version
        self.modules = modules

    def get_table(self, module, table):
        """
        :param module: The name of the module to fetch the table from.
        :param table: The name of the table to retrieve.
        :return: The table as pandas DataFrame.
        """
        table_path = path.join(self.database_path, self.modules[module], self.tables[table])
        # noinspection PyTypeChecker
        return pd.read_csv(table_path, delimiter=self.delimiter, encoding=self.encoding)

    def get_analyses(self):
        return self.get_table("hospital", "analyses")

    def get_patients(self):
        return self.get_table("hospital", "patients")

    def get_diagnoses(self):
        return self.get_table("hospital", "diagnoses")

    def get_diagnoses_lookup(self):
        return self.get_table("hospital", "diagnoses_lookup")

    def get_analyses_lookup(self):
        return self.get_table("hospital", "analyses_lookup")

    @staticmethod
    def create_config_file(config_path):
        """
        Creates a configuration file for the data manager. The parameters are
        Initialized with default values for MIMIC-IV.

        Parameters:
        - config_path (str): Path to save the configuration file.
        """
        default_config = {
            'database_root': '',
        }

        with open(config_path, 'w') as config_file:
            json.dump(default_config, config_file, indent=4, sort_keys=True)
        print(f"Default configuration file for MIMIC-IV dataset created at {config_path}.")
        print(f"Make sure to change the path to point at the root of the mimic dataset.")

    @staticmethod
    def from_config_file(config_path):
        """
        Loads configuration from a file and initializes the MimicDataset.
        :param config_path: Path to the configuration file.
        :returns: `MimicDataset` instance
        """
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)

        return MimicDataset(config)
