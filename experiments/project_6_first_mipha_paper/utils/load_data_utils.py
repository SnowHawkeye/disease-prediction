import os

from sklearn.impute import SimpleImputer

from features.mimic.extract_lab_records import save_to_file
from models.mipha.utils.data_processing import mask_train_test_split
from src.features.mimic.extract_lab_records import load_pickle
from src.models.mipha.data_sources.mimic.demographics_datasource import DemographicsDataSource
from src.models.mipha.data_sources.mimic.patient_record_datasource import PatientRecordDataSource
from src.models.mipha.data_sources.mimic.record_to_matrix_conversion import create_mask_from_records, \
    create_label_matrix


def make_simple_imputer(strategy="mean"):
    return SimpleImputer(keep_empty_features=True, strategy=strategy)


def make_data_sources(
        root_dir,
        bgp_string,
        lab_categories,
        disease_identifier,
        test_size=0.2,
        random_seed=None,
        imputer="auto"
):
    print("Loading records...")
    demographics_records, ecg_records, lab_records, labeled_records = load_records(
        root_dir=root_dir, bgp_string=bgp_string, lab_categories=lab_categories, disease_identifier=disease_identifier
    )

    print("Creating mask...")
    mask = create_mask_from_records(labeled_records)  # mapping of patient IDs and timestamps for learning matrices

    print("Creating train and test masks...")
    mask_train, mask_test = mask_train_test_split(mask=mask, test_size=test_size, random_seed=random_seed)

    print("Creating label matrices...")
    labels_train = create_label_matrix(labels_dict=labeled_records, mask=mask_train)
    labels_test = create_label_matrix(labels_dict=labeled_records, mask=mask_test)

    print("Creating laboratory data sources and splitting train and test sets...")
    # lab_data_sources is a dictionary because of all the categories
    lab_data_sources = make_lab_data_sources(lab_records, mask_train, mask_test, imputer)

    print("Creating ECG data sources...")
    ecg_data_source_train, ecg_data_source_test = make_ecg_data_sources(ecg_records, mask_train, mask_test, imputer)

    print("Creating 2D demographics data sources...")
    demographics_2d_data_source_train, demographics_2d_data_source_test \
        = make_demographics_data_sources_2d(demographics_records, mask_train, mask_test)

    print("Creating 3D demographics data sources...")
    # Assuming that all time series have the same number of time steps
    n_timesteps = ecg_data_source_train.data.shape[1]

    demographics_data_source_3d_test, demographics_data_source_3d_train \
        = make_demographics_data_sources_3d(demographics_records, mask_test, mask_train, n_timesteps)

    data_sources = {
        "ecg": (ecg_data_source_train, ecg_data_source_test),
        "demographics_2d": (demographics_2d_data_source_train, demographics_2d_data_source_test),
        "demographics_3d": (demographics_data_source_3d_train, demographics_data_source_3d_test),
        "labels": (labels_train, labels_test)
    }

    data_sources.update(lab_data_sources)  # add lab data sources to the dictionary

    return data_sources


def load_records(root_dir, bgp_string, lab_categories, disease_identifier):
    ecg_records = load_ecg_records(root_dir=root_dir, bgp_string=bgp_string)
    demographics_records = load_demographics_records(root_dir=root_dir, bgp_string=bgp_string)
    labeled_records = load_labeled_records(disease=disease_identifier, root_dir=root_dir, bgp_string=bgp_string)

    lab_records = {}
    for lab_category in lab_categories:
        lab_records[lab_category] = load_lab_records(category=lab_category, root_dir=root_dir, bgp_string=bgp_string)

    return demographics_records, ecg_records, lab_records, labeled_records


def load_lab_records(category, root_dir, bgp_string):
    """
    Load a rolling lab record. Expected path structure defined in src/features/scripts/config/create_config_files.ipynb
    :param category: name of the laboratory data category
    :param root_dir: root_directory ("*/categorized_analyses")
    :param bgp_string: backward window, gap and prediction window as a string (formatted like in the paths)
    :return: the rolling lab record loaded from the file
    """

    return load_pickle(
        os.path.join(root_dir, "lab", category, bgp_string, f"rolling_lab_records_{category}_{bgp_string}.pkl")
    )


def load_ecg_records(root_dir, bgp_string):
    return load_pickle(
        os.path.join(root_dir, "ecg", bgp_string, f"rolling_ecg_records_{bgp_string}.pkl")
    )


def load_demographics_records(root_dir, bgp_string):
    return load_pickle(
        os.path.join(root_dir, "demographics", bgp_string, f"demographics_records_{bgp_string}.pkl")
    )


def load_labeled_records(disease, root_dir, bgp_string):
    return load_pickle(
        os.path.join(root_dir, "labels", disease, bgp_string, f"labeled_lab_records_{disease}_{bgp_string}.pkl")
    )


def make_lab_data_sources(lab_records, mask_train, mask_test, imputer="auto"):
    if imputer == "auto":
        imputer = make_simple_imputer()

    lab_data_sources = {}

    for category, records in lab_records.items():
        print(f"Creating data sources for category {category}...")

        data_source_train = PatientRecordDataSource(
            data_type="laboratory",
            name=f"{category}_lab_data_source_train",
            data=records,
            mask=mask_train,
        )
        data_source_train.impute_data(imputer)

        data_source_test = PatientRecordDataSource(
            data_type="laboratory",
            name=f"{category}_lab_data_source_test",
            data=records,
            mask=mask_test,
        )
        data_source_test.impute_data(imputer)

        lab_data_sources[category] = (data_source_train, data_source_test)
        print(f"Created lab data sources for train and test "
              f"with shapes: {data_source_train.data.shape}, {data_source_test.data.shape})")

    return lab_data_sources


def make_ecg_data_sources(ecg_records, mask_train, mask_test, imputer="auto"):
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


def make_demographics_data_sources_2d(demographics_records, mask_train, mask_test):
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


def load_data_sources(
        root_dir,
        bgp_string,
        lab_categories,
        disease_identifier,
        save_to=None,
        random_seed=None,
        imputer="auto"
):

    print("Loading data sources...")
    if not os.path.exists(save_to) or save_to is None:
        print(f"No pre-processed data sources have been found at {save_to}. Loading data sources from files...")
        data_sources = make_data_sources(
            root_dir=root_dir,
            bgp_string=bgp_string,
            lab_categories=lab_categories,
            disease_identifier=disease_identifier,
            test_size=0.2,
            random_seed=random_seed,
            imputer=imputer
        )

        print("Saving data sources...")
        save_to_file(data_sources, save_to)

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
