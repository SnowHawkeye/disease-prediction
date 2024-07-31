import json

import pytest

from code.datasets.mimic_dataset import MimicDataset


@pytest.fixture
def sample_config(tmp_path):
    config = {
        'load_params': {
            'database_path': str(tmp_path),
            'delimiter': ',',
            'encoding': 'utf-8',
        },
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
    config_path = tmp_path / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    return config, config_path


def test_initialization(sample_config):
    config, _ = sample_config
    dataset = MimicDataset(config)

    assert dataset.database_path == config['load_params']['database_path']
    assert dataset.delimiter == config['load_params']['delimiter']
    assert dataset.encoding == config['load_params']['encoding']
    assert dataset.column_aliases == config['column_aliases']
    assert dataset.tables == config['tables']
    assert dataset.modules == config['modules']


def test_create_config_file(tmp_path):
    config_path = tmp_path / "default_config.json"
    MimicDataset.create_config_file(config_path)

    with open(config_path, 'r') as f:
        config = json.load(f)

    assert config['load_params']['database_path'] == ''
    assert config['load_params']['delimiter'] == ','
    assert config['load_params']['encoding'] == 'utf-8'
    assert config['column_aliases'] == {
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
    assert config['tables'] == {
        "analyses_lookup": "d_labitems.csv.gz",
        "diagnoses_lookup": "d_icd_diagnoses.csv.gz",
        "patients": "patients.csv.gz",
        "diagnoses": "diagnoses_icd.csv.gz",
        "analyses": "labevents.csv.gz"
    }
    assert config['modules'] == {
        "hospital": "hosp"
    }


def test_from_config_file(sample_config):
    _, config_path = sample_config
    dataset = MimicDataset.from_config_file(config_path)

    assert isinstance(dataset, MimicDataset)
    assert dataset.database_path == str(sample_config[0]['load_params']['database_path'])
    assert dataset.delimiter == sample_config[0]['load_params']['delimiter']
    assert dataset.encoding == sample_config[0]['load_params']['encoding']
    assert dataset.column_aliases == sample_config[0]['column_aliases']
    assert dataset.tables == sample_config[0]['tables']
    assert dataset.modules == sample_config[0]['modules']
