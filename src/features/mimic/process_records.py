import pandas as pd
from tqdm import tqdm


def make_rolling_records(patient_records, time_unit: str, backward_window: int, observation_dates=None):
    """
    Processes a dictionary of patient records creating sub-DataFrames for each time point (observation date),
    containing data from the previous `backward_window` units of time.
    If not provided, observation dates correspond to each record date, after resampling at the frequency given by `time_unit`.

    Parameters:
        :param patient_records: A dictionary where keys are patient IDs and values are DataFrames.
        :param time_unit: Base time unit for the output's indexes. Possible values are: {'day', 'week', 'month', 'year'}
        :param backward_window: Number of units to look back for each time point.
        :param observation_dates: A dictionary where keys are patient IDs and values are lists of timestamps, corresponding to the observation dates for each patient.

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

    for key, df in tqdm(patient_records.items(), desc="Creating rolling records for patients"):
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


def label_records(patient_rolling_records, gap_days, prediction_window_days, positive_diagnoses,
                  diagnoses_table, admissions_table):
    """
    Using the given patient rolling records, returns a dictionary associating each observation date for each patient to a label.
    The label is computed by checking whether there is a "positive diagnosis" in the prediction window after the observation date.
    If an observation date (+gap) is later than the earliest positive diagnosis, it is omitted from the output
    (since the goal is to predict the onset of a disease, it is considered pointless to use data after the earliest diagnosis).

    :param patient_rolling_records: A dictionary where the keys are patient IDs and values are either sub-dictionaries where observation dates are keys, or directly lists of observation dates.
    :param gap_days: The gap period between the observation date and the prediction window, in days.
    :param prediction_window_days: The prediction window, in days.
    :param positive_diagnoses: A list of diagnoses considered positive.
    :param diagnoses_table: The table of all diagnoses.
    :param admissions_table: The table of all admissions, used to retrieve discharge times (corresponding to diagnosis times).
    :return: A dictionary where the keys are patient IDs, and values are sub-dictionaries where keys are observation dates, and values are binary labels (0 or 1).
    """

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

    for patient_id, rolling_records in tqdm(patient_rolling_records.items(), desc="Labeling tables for patients"):
        if patient_id not in grouped_diagnoses.groups:
            patient_not_found_count += 1
            continue

        labels_for_patient = {}  # dictionary containing the labels for each observation date
        timed_diagnoses = (grouped_diagnoses.get_group(patient_id)  # merging to include discharge_time = diagnosis time
                           .merge(grouped_admissions.get_group(patient_id), on="admission_id"))

        # finding the earliest diagnosis of interest, if it exists
        filtered_timed_diagnoses = timed_diagnoses[timed_diagnoses["diagnosis_code"].isin(positive_diagnoses)]
        earliest_date = filtered_timed_diagnoses["discharge_time"].min()  # NaT if no such date exists

        if type(rolling_records) is dict:
            observation_dates = rolling_records.keys()
        elif type(rolling_records) is list:
            observation_dates = rolling_records
        else:
            raise TypeError("rolling_records must be a dictionary or a list")

        for observation_date in observation_dates:
            if pd.isna(earliest_date):  # if the patient is negative
                label = 0  # the label will always be 0

            else:  # if the patient is positive
                # if the observation date is later than the earliest diagnosis
                # observation_date + gap = "effective" observation date
                if observation_date + gap_delta > earliest_date:
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
