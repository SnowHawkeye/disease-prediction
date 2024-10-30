import os
from os import path

from sklearn.impute import SimpleImputer

from src.models.mipha.data_sources.mimic.demographics_datasource import DemographicsDataSource
from src.models.mipha.data_sources.mimic.patient_record_datasource import PatientRecordDataSource
from src.models.mipha.data_sources.mimic.record_to_matrix_conversion import create_mask_from_records, \
    create_label_matrix
from src.features.mimic.extract_lab_records import save_pickle, load_pickle


def make_simple_imputer(strategy="mean"):
    return SimpleImputer(keep_empty_features=True, strategy=strategy)


# hardcoded file names should not be a problem for this experiment since data is generated to follow this pattern
# might want to change in the future
lab_records_file = "rolling_lab_records.pkl"
ecg_records_file = "rolling_ecg_records.pkl"
demographics_records_file = "demographics_records.pkl"
labels_file = "labeled_lab_records.pkl"


def make_data_sources(data_root, test_size=0.2, random_seed=None, imputer="auto"):
    print("Loading records...")
    demographics_records, ecg_records, lab_records, labeled_records = load_records(data_root)

    print("Creating mask...")
    mask = create_mask_from_records(labeled_records)  # mapping of patient IDs and timestamps for learning matrices

    print("Creating label matrix...")
    labels = create_label_matrix(labels_dict=labeled_records, mask=mask)

    print("Creating laboratory data sources and splitting train and test sets...")
    lab_data_source_train, lab_data_source_test, labels_train, labels_test, mask_train, mask_test \
        = make_lab_data_sources(lab_records, labels, random_seed, test_size, mask, imputer)

    print("Creating ECG data sources...")
    ecg_data_source_train, ecg_data_source_test = make_ecg_data_sources(ecg_records, mask_test, mask_train, imputer)

    print("Creating 2D demographics data sources...")
    demographics_2d_data_source_train, demographics_2d_data_source_test \
        = make_demographics_data_sources_2d(demographics_records, mask_test, mask_train)

    print("Creating 3D demographics data sources...")
    n_timesteps = lab_data_source_train.data.shape[1]

    demographics_data_source_3d_test, demographics_data_source_3d_train \
        = make_demographics_data_sources_3d(demographics_records, mask_test, mask_train, n_timesteps)

    return {
        "lab_data_sources": (lab_data_source_train, lab_data_source_test),
        "ecg_data_sources": (ecg_data_source_train, ecg_data_source_test),
        "demographics_data_sources_2d": (demographics_2d_data_source_train, demographics_2d_data_source_test),
        "demographics_data_sources_3d": (demographics_data_source_3d_train, demographics_data_source_3d_test),
        "labels": (labels_train, labels_test)
    }


def load_records(data_root):
    lab_records = load_pickle(path.join(data_root, lab_records_file))
    ecg_records = load_pickle(path.join(data_root, ecg_records_file))
    demographics_records = load_pickle(path.join(data_root, demographics_records_file))
    labeled_records = load_pickle(path.join(data_root, labels_file))
    return demographics_records, ecg_records, lab_records, labeled_records


def make_lab_data_sources(lab_records, labels, random_seed, test_size, mask, imputer="auto"):
    if imputer == "auto":
        imputer = make_simple_imputer()

    lab_data_source = PatientRecordDataSource(
        data_type="laboratory",
        name="lab_data_source",
        data=lab_records,
        mask=mask,
    )

    print("Imputing data...")
    lab_data_source.impute_data(imputer)

    print("Splitting training and test set...")
    lab_data_source_train, lab_data_source_test, labels_train, labels_test = lab_data_source.split_train_test(
        labels=labels, test_size=test_size, random_seed=random_seed
    )
    print(f"Split lab data source (shape: {lab_data_source.data.shape}) "
          f"into train and test sets (shapes: {lab_data_source_train.data.shape}, {lab_data_source_test.data.shape})")

    print("Creating training and test masks...")
    mask_train = lab_data_source_train.mask
    mask_test = lab_data_source_test.mask
    return lab_data_source_train, lab_data_source_test, labels_train, labels_test, mask_train, mask_test


def make_ecg_data_sources(ecg_records, mask_test, mask_train, imputer="auto"):
    if imputer == "auto":
        imputer = make_simple_imputer()

    ecg_data_source_train = PatientRecordDataSource(
        data_type="ecg",
        name="ecg_data_source_train",
        data=ecg_records,
        mask=mask_train
    )
    ecg_data_source_train.impute_data(imputer)
    ecg_data_source_test = PatientRecordDataSource(
        data_type="ecg",
        name="ecg_data_source_test",
        data=ecg_records,
        mask=mask_test
    )
    ecg_data_source_test.impute_data(imputer)
    print(f"Created ECG data sources for train and test "
          f"with shapes: {ecg_data_source_train.data.shape}, {ecg_data_source_test.data.shape})")

    return ecg_data_source_train, ecg_data_source_test


def make_demographics_data_sources_2d(demographics_records, mask_test, mask_train):
    demographics_data_source_2d_train = DemographicsDataSource(
        data_type="demographics",
        name="demographics_data_source_2d_train",
        data=demographics_records,
        mask=mask_train
    )
    demographics_data_source_2d_train.flatten_data()
    demographics_data_source_2d_test = DemographicsDataSource(
        data_type="demographics",
        name="demographics_data_source_2d_test",
        data=demographics_records,
        mask=mask_test
    )
    demographics_data_source_2d_test.flatten_data()
    print(f"Created demographics (2D) data sources for train and test "
          f"with shapes: {demographics_data_source_2d_train.data.shape}, {demographics_data_source_2d_test.data.shape})")

    return demographics_data_source_2d_train, demographics_data_source_2d_test


def make_demographics_data_sources_3d(demographics_records, mask_test, mask_train, n_timesteps):
    demographics_data_source_3d_train = DemographicsDataSource(
        data_type="demographics",
        name="demographics_data_source_3d_train",
        data=demographics_records,
        mask=mask_train
    )
    demographics_data_source_3d_train.extend_timesteps(n_timesteps)
    demographics_data_source_3d_test = DemographicsDataSource(
        data_type="demographics",
        name="demographics_data_source_3d_test",
        data=demographics_records,
        mask=mask_test
    )
    demographics_data_source_3d_test.extend_timesteps(n_timesteps)
    print(f"Created demographics (3D) data sources for train and test "
          f"with shapes: {demographics_data_source_3d_train.data.shape}, {demographics_data_source_3d_test.data.shape})")
    return demographics_data_source_3d_test, demographics_data_source_3d_train


def load_data_sources(data_root, save_to=None, random_seed=None, imputer="auto"):
    print("Loading data sources...")
    if not os.path.exists(save_to) or save_to is None:
        print("No pre-processed data sources have been found at. Loading data sources from root...")
        data_sources = make_data_sources(data_root=data_root, test_size=0.2, random_seed=random_seed, imputer=imputer)

        print("Saving data sources...")
        save_pickle(data_sources, save_to)

    else:  # if the datasource setup has already been saved
        print("Pre-processed data sources have been found. Loading data sources from pickle...")
        data_sources = load_pickle(save_to)
    return data_sources


def setup_data_sources(data_sources, kept_data_sources):
    data_sources_train = []
    data_sources_test = []
    for kept_datasource in kept_data_sources:
        data_train, data_test = data_sources[kept_datasource]
        data_sources_train.append(data_train)
        data_sources_test.append(data_test)

    labels_train, labels_test = data_sources["labels"]

    return data_sources_train, data_sources_test, labels_train, labels_test
