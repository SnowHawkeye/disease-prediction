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


def extract_lab_records(analyses, freq='h', aggfunc='mean'):
    """
    Extracts laboratory records for each patient and returns a dictionary where
    keys are patient IDs and values are DataFrames with pivoted laboratory records.

    :param analyses: DataFrame containing all analyses.
    :param freq: Frequency of the laboratory records (cf. https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases)
    :param aggfunc: Function used to aggregate values. Refer to https://pandas.pydata.org/docs/reference/api/pandas.pivot_table.html
    :return: A dict with patient IDs as keys and a DataFrame of laboratory data as values.
    """
    print("Extracting patient laboratory records...")

    # Clean and prepare the data
    analyses_clean = analyses.dropna(subset=["analysis_time"])  # Remove rows with missing time
    analyses_clean["analysis_time"] = pd.to_datetime(analyses_clean["analysis_time"])  # Convert to datetime

    # Group by patient ID and process each group
    patient_dataframes = {}
    grouped = analyses_clean.groupby("patient_id")

    for patient_id, group_data in tqdm(grouped, desc="Extracting patient data"):
        pivot_data = group_data.pivot_table(
            index=pd.Grouper(key="analysis_time", freq=freq),  # Group analyses by hour
            columns="analysis_id",  # Use analysis_id as columns
            values="analysis_value",  # Values to aggregate
            aggfunc=aggfunc  # Aggregate function
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


def make_rolling_records(patient_lab_records, time_unit: str, backward_window: int):
    """
    Processes a dictionary of patient lab records creating sub-DataFrames for each time point (observation date),
    containing data from the previous `backward_window` units of time.
    Observation dates correspond to each record date, after resampling at the frequency given by `time_unit`.

    Parameters:
        :param patient_lab_records: A dictionary where keys are identifiers and values are DataFrames.
        :param time_unit: Base time unit for the output's indexes. Possible values are: {'day', 'week', 'month', 'year'}
        :param backward_window: Number of units to look back for each time point.

    Returns:
        dict: A dictionary of dictionaries, where each inner dictionary contains sub-DataFrames for each observation date.
    """
    offsets = {  # pandas offset aliases
        'day': 'D',
        'week': 'W',
        'month': 'ME',
        'year': 'YE',
    }

    deltas = {  # pandas offset aliases
        'day': '1 days',
        'week': '7 days',
        'month': '31 days',
        'year': '366 days',
        # months and years are set to their upper bounds to fix inconsistencies when creating backwards windows
    }

    result = {}

    for key, df in tqdm(patient_lab_records.items(), desc="Creating rolling records for patients"):
        # Resample the dataframe according to the specified frequency
        resampled = df.resample(offsets[time_unit]).mean().asfreq(offsets[time_unit]).dropna(how='all')

        # Create a dictionary to store sub-DataFrames for each time point
        sub_dfs = {}

        # Iterate over the resampled index
        for date in resampled.index:
            # Calculate the start date for the window
            one_time_unit = pd.Timedelta(deltas[time_unit])
            start_date = date - one_time_unit * backward_window

            # Filter the original DataFrame for the given window
            sub_df = resampled[(resampled.index > start_date) & (resampled.index <= date)].copy()

            # Create entries for time units with missing data
            # +1 unit on start date to have B elements in the dataframe
            complete_date_range = pd.date_range(start=start_date + one_time_unit, end=date, freq=offsets[time_unit])

            full_df = pd.DataFrame(index=complete_date_range)
            full_df = full_df.merge(sub_df, left_index=True, right_index=True, how='left')

            # Add the sub-DataFrame to the dictionary
            sub_dfs[date] = full_df

        # Store the dictionary of sub-DataFrames in the result dictionary
        result[key] = sub_dfs

    return result


def label_lab_records(patient_rolling_lab_records, gap_days, prediction_window_days, positive_diagnoses,
                      diagnoses_table,
                      admissions_table):
    labels = {}

    print("Converting times to timestamps...")
    admissions_copy = admissions_table.copy()
    admissions_copy["discharge_time"] = pd.to_datetime(admissions_copy["discharge_time"], errors='coerce')
    admissions_copy.dropna(subset="discharge_time", inplace=True)

    print("Preparing tables...")
    grouped_diagnoses = diagnoses_table.groupby("patient_id")
    grouped_admissions = admissions_copy.groupby("patient_id")

    gap_delta = pd.Timedelta(f"{gap_days} days")
    prediction_window_delta = pd.Timedelta(f"{prediction_window_days} days")

    patient_not_found_count = 0

    for patient_id, rolling_records in tqdm(patient_rolling_lab_records.items(), desc="Labeling tables for patients"):
        if patient_id not in grouped_diagnoses.groups:
            patient_not_found_count += 1
            continue

        labels_for_patient = {}  # dictionary containing the labels for each observation date
        timed_diagnoses = (grouped_diagnoses.get_group(patient_id)  # merging to include discharge_time = diagnosis time
                           .merge(grouped_admissions.get_group(patient_id), on="admission_id"))

        # finding the earliest diagnosis of interest, if it exists
        filtered_timed_diagnoses = timed_diagnoses[timed_diagnoses["diagnosis_code"].isin(positive_diagnoses)]
        earliest_date = filtered_timed_diagnoses["discharge_time"].min()  # NaT if no such date exists

        for observation_date, df in rolling_records.items():
            if pd.isna(earliest_date):  # if the patient is negative
                label = 0  # the label will always be 0

            else:  # if the patient is positive
                # if the observation date is later than the earliest diagnosis
                # observation_date + gap = "effective" observation date
                if observation_date + gap_delta> earliest_date:
                    continue  # skip this observation date, do not add a label
                else:  # the observation date is before the earliest diagnosis
                    prediction_window_start = observation_date + gap_delta
                    prediction_window_end = observation_date + gap_delta + prediction_window_delta

                    label = 1 if prediction_window_start <= earliest_date <= prediction_window_end else 0

            labels_for_patient[observation_date] = label

        labels[patient_id] = labels_for_patient

    print(
        f"Records for {len(labels)} patients labeled. {patient_not_found_count} patients not found in diagnoses table."
    )

    return labels


def save_pickle(dataframes, filename):
    """
    :param dataframes: provided as a dictionary where keys are patient IDs
    :param filename: file to save to
    """
    print(f"Saving data to {filename}...")
    with open(filename, "wb") as handle:
        pickle.dump(dataframes, handle)
    print(f"Data successfully saved to {filename}")


def load_pickle(filename):
    """
    :param filename: file to load data from
    """
    print(f"Loading data from {filename}...")
    with open(filename, "rb") as file:
        result = pickle.load(file)
        print(f"Data successfully loaded from {filename}")
        return result


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
