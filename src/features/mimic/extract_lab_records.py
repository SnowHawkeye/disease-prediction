import _pickle as pickle
import os
import tempfile
import zipfile

import numpy as np
from tqdm import tqdm
import pandas as pd

# refer to https://pandas.pydata.org/pandas-docs/stable/user_guide/copy_on_write.html#copy-on-write
# in short, when assigning a slice of a dataframe to a variable, modifying the slice no longer modifies the dataframe
pd.options.mode.copy_on_write = True


def extract_lab_records(analyses, cols):
    """
    Extracts laboratory records for each patient and returns a dictionary where
    keys are patient IDs and values are DataFrames with pivoted laboratory records.

    :param analyses: DataFrame containing all analyses.
    :param cols: Mapping of column names to "normalized" names.
    :return: A dict with patient IDs as keys and a DataFrame of laboratory data as values.
    """
    print("Extracting patient laboratory records...")

    # Clean and prepare the data
    analyses_clean = analyses.dropna(subset=[cols["analysis_time"]])  # Remove rows with missing time
    analyses_clean[cols["analysis_time"]] = pd.to_datetime(analyses_clean[cols["analysis_time"]])  # Convert to datetime

    # Group by patient ID and process each group
    patient_dataframes = {}
    grouped = analyses_clean.groupby(cols["patient_id"])

    for patient_id, group_data in tqdm(grouped, desc="Extracting patient data"):
        pivot_data = group_data.pivot_table(
            index=pd.Grouper(key=cols["analysis_time"], freq='h'),  # Group analyses by hour
            columns=cols["analysis_id"],  # Use analysis_id as columns
            values=cols["analysis_value"],  # Values to aggregate
            aggfunc='mean'  # Aggregate function
        )
        if not pivot_data.empty:
            patient_dataframes[int(patient_id)] = pivot_data

    print(f"{len(patient_dataframes)} non-empty patient records extracted")
    return patient_dataframes


def filter_lab_records(patient_lab_records, analyses_ids):
    """
    For each patient record in the given dict, keep only the analyses columns corresponding to the given in analyses_ids.
    If the columns do not exist, create new columns filled with NaNs.
    :param patient_lab_records: A dict with patient IDs as keys and a DataFrame of laboratory data as values.
    :param analyses_ids: The analyses to keep.
    :return: The filtered laboratory records, with news nan-filled columns when needed.
    """
    filtered_dict = {
        key: df.reindex(columns=analyses_ids, fill_value=np.nan)
        for key, df in tqdm(patient_lab_records.items())
    }
    return filtered_dict


def save_patient_records_to_pickle(dataframes, filename):
    """
    :param dataframes: provided as a dictionary where keys are patient IDs
    :param filename: file to save to
    """
    print(f"Saving patient records to {filename}...")
    with open(filename, "wb") as handle:
        pickle.dump(dataframes, handle)
    print(f"Patient records successfully saved to {filename}")


def load_patient_records_from_pickle(filename):
    """
    :param filename: file to load data from
    """
    with open(filename, "rb") as file:
        return pickle.load(file)


def save_patient_records_to_parquet_archive(data, archive_name='data.zip'):
    # Create a temporary directory for storing Parquet files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save each DataFrame to a separate Parquet file
        for key, df in data.items():
            file_path = f'{temp_dir}/{key}.parquet'
            df.to_parquet(file_path, index=False)

        # Create a ZIP archive of the Parquet files
        with zipfile.ZipFile(archive_name, 'w') as archive:
            for file_name in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, file_name)
                archive.write(file_path, file_name)


def load_patient_records_from_parquet_archive(archive_name='data.zip'):
    # Create a temporary directory for extracting Parquet files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract files from the ZIP archive
        with zipfile.ZipFile(archive_name, 'r') as archive:
            archive.extractall(temp_dir)

        # Load DataFrames from the extracted Parquet files
        data = {}
        for file_name in tqdm(os.listdir(temp_dir)):
            if file_name.endswith('.parquet'):
                key = file_name.replace('.parquet', '')
                file_path = os.path.join(temp_dir, file_name)
                data[key] = pd.read_parquet(file_path)

    return data
