import numpy as np
from tqdm import tqdm


def create_mask_from_records(patient_records):
    """
    Creates a list of (patient_id, timestamp) tuples from the given records.
    Patients are kept only if they have at least one timestamp (observation date).
    The purpose of this mask is to be aligned with the matrix data, so that patients and observation dates
    can be retraced from their matrices.
    :param patient_records: A dictionary where keys are patient IDs, and values are dictionaries
    where keys are the timestamps (observation dates). The values of these sub-dictionaries will be ignored. This format
    corresponds to the format returned by the records extraction scripts.
    """
    mask = []
    for patient_id, timestamps in tqdm(patient_records.items(), desc="Creating mask from records"):
        for timestamp in timestamps:
            mask.append((patient_id, timestamp))
    return mask


def create_label_matrix(labels_dict, mask):
    """
    Creates a matrix using the labels provided.
    :param labels_dict: A dictionary where the keys are patient IDs, and values are dictionaries where keys are timestamps,
    and values are labels.
    :param mask: A list of (patient_id, timestamp) tuples indicating what each row of the output corresponds to, as returned by `create_mask`.
    :return: A matrix containing the labels for each patient. The order is given by the mask.
    """
    labels = []
    for patient_id, timestamp in tqdm(mask, desc="Creating label matrix"):
        labels.append(labels_dict[patient_id][timestamp])

    return np.array(labels)


def create_learning_matrix(patient_records, mask, fill_value=np.NaN):
    """
    Creates a matrix using the records provided. Unlike the given records, the matrix is a numpy array, suitable for machine learning.
    :param patient_records: A dictionary where the keys are patient IDs, and values are dictionaries where keys are timestamps (observation dates),
    and values contain the data as dataframes. ⚠️ All dataframes are assumed to have the same shape.
    :param mask: A list of (patient_id, timestamp) tuples indicating what each row of the output corresponds to, as returned by `create_mask`.
    :param fill_value: The value to use when data is missing (default is np.NaN).
    :return: A matrix containing the labels for each patient. The order is given by the mask.
    """
    learning_matrix = []

    # Determine the shape of the data (assuming all dataframes have the same shape)
    sample_shape = None
    for patient_id, timestamp in mask:
        if patient_id in patient_records and timestamp in patient_records[patient_id]:
            sample_shape = patient_records[patient_id][timestamp].shape
            break

    if sample_shape is None:
        raise ValueError(
            "Could not determine the shape of the data. Check that the patient_records contain valid entries. "
            "This may happen if the patient records and the mask have no overlapping keys."
        )

    for patient_id, timestamp in tqdm(mask, desc="Creating learning matrix"):
        if patient_id in patient_records and timestamp in patient_records[patient_id]:
            # If both patient_id and timestamp exist, add the data
            data = patient_records[patient_id][timestamp].to_numpy()
        else:
            # Fill with the specified fill_value if data is missing
            data = np.full(sample_shape, fill_value)

        learning_matrix.append(data)

    return np.array(learning_matrix)
