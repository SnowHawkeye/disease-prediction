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
        self.config = config
        self.database_path = config['load_params'].get('database_path', '')
        self.delimiter = config['load_params'].get('delimiter', ',')
        self.encoding = config['load_params'].get('encoding', 'utf-8')
        self.column_aliases = config.get('column_aliases', {})
        self.tables = config.get('tables', {})
        self.modules = config.get('modules', {})

    def get_table(self, module, table):
        """
        :param module: The name of the module to fetch the table from.
        :param table: The name of the table to retrieve.
        :return: The table as pandas DataFrame.
        """
        table_path = path.join(self.database_path, self.modules[module], self.tables[table])
        # noinspection PyTypeChecker
        return pd.read_csv(table_path, delimiter=self.delimiter, encoding=self.encoding)

    @staticmethod
    def create_config_file(config_path):
        """
        Creates a configuration file for the data manager. The parameters are
        Initialized with default values for MIMIC-IV.

        Parameters:
        - config_path (str): Path to save the configuration file.
        """
        default_config = {
            'load_params': {
                'database_path': '',
                'delimiter': ',',
                'encoding': 'utf-8',
            },
            # keys are the "normalized" names, values are the corresponding names in the database
            'column_aliases': {
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
            },
            'tables': {
                "analyses_lookup": "d_labitems.csv.gz",
                "diagnoses_lookup": "d_icd_diagnoses.csv.gz",
                "patients": "patients.csv.gz",
                "diagnoses": "diagnoses_icd.csv.gz",
                "analyses": "labevents.csv.gz"
            },
            'modules': {
                "hospital": "hosp"
            }
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
