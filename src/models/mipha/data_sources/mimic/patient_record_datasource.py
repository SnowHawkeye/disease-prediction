import warnings

from mipha.framework import DataSource
from models.mipha.data_sources.mimic.record_to_matrix_conversion import create_learning_matrix, create_mask_from_records
from models.mipha.utils.data_processing import impute_data, scale_time_series_data_train, scale_time_series_data_test, \
    resample_3d_data
import numpy as np
from models.mipha.utils.data_processing import mask_train_test_split


class PatientRecordDataSource(DataSource):
    def __init__(self, data_type, name, data, mask=None):
        """
        Represents a data source with a specific type, name, and data.
        This data source is suitable for data extracted as dictionaries where keys are patient IDs, 
        and values are sub-dictionaries. The keys of the sub-dictionaries are timestamps (observation dates) 
        and their values are dataframes containing the data (usually time series).
        It can also accept pre-formatted 3D numpy arrays.

        :param data_type: The type of data contained in the data source. 
        :type data_type: str

        :param name: The name of the data source.
        :type name: str

        :param data: The actual data of the data source. Refer to the description above for the format.
                     If it is not already a numpy array, it will be converted to a 3D numpy array.

        :param mask: The mask associated with the data source, given as a list of (patient_id, timestamp) tuples. 
                     When given, the numpy array associated with the `data` parameter will match this mask. 
                     If not, a mask will be created from the data directly if the data is provided as a dictionary (formatted as above).
        """

        super().__init__(data_type=data_type, name=name, data=data)

        self.mask = mask
        if isinstance(data, dict):  # Only call create_learning_matrix if data is not already a numpy array
            if self.mask is None:  # Create mask from dict if it was not provided
                self.mask = create_mask_from_records(patient_records=data)
            self.data = create_learning_matrix(patient_records=data, mask=self.mask)
        else:
            self.data = data

        if self.mask is None:  # if no mask was created (data was not provided as dict)
            warnings.warn(
                "Mask could not be created from the data and was set to None. In particular, the split_train_test method will not work."
            )

    def impute_data(self, imputer):
        """
        Impute this data source's data using the provided imputer.
        ⚠️Imputation is performed on a per-matrix basis, meaning that values are imputed only using the considered history.
        Because of this, imputation can be performed on training and test data indifferently.
        """
        self.data = impute_data(data=self.data, imputer=imputer)

    def split_train_test(self, labels, test_size=0.2, random_seed=None, resampler=None, scaler=None):
        """
        Split the data source's data into two new data sources: one for training and one for testing.
        Optionally scales the data.

        :param labels: The labels associated with the data source.

        :param test_size: Proportion of patient IDs to include in the test split (default is 0.2).
        :type test_size: float

        :param random_seed: Seed for the random number generator for reproducibility (default is None).
        :type random_seed: int or None

        :param resampler: The resampler used to balance the data, only applied to training data. Defaults to None. If no resampler is provided, the data isn't resampled.

        :param scaler: The scaler used to scale the data. Defaults to None. If no scaler is provided, the data isn't scaled.

        :return: Two new PatientRecordDatasource instances for training and testing and their associated labels. In order: train_datasource, test_datasource, train_labels, test_labels
        :rtype: tuple(PatientRecordDatasource, PatientRecordDatasource, numpy.ndarray, numpy.ndarray)
        """

        # Split the mask into training and testing masks
        train_mask, test_mask = mask_train_test_split(self.mask, test_size=test_size, random_seed=random_seed)

        # Split the data according to the new masks
        train_data, test_data, train_labels, test_labels = [], [], [], []
        for (patient_id, timestamp), record, label in zip(self.mask, self.data, labels):
            if (patient_id, timestamp) in train_mask:
                train_data.append(record)
                train_labels.append(label)
            else:
                test_data.append(record)
                test_labels.append(label)

        # Convert lists back into numpy arrays
        train_data = np.array(train_data)
        test_data = np.array(test_data)
        train_labels = np.array(train_labels)

        # Create new data sources for training and testing
        train_datasource = PatientRecordDataSource(data_type=self.data_type,
                                                   name=f"{self.name}_train",
                                                   data=train_data,
                                                   mask=train_mask)

        test_datasource = PatientRecordDataSource(data_type=self.data_type,
                                                  name=f"{self.name}_test",
                                                  data=test_data,
                                                  mask=test_mask)

        if resampler is not None:
            print(f"Resampling training data. Current shape is {train_datasource.data.shape}.")
            train_datasource.data, train_labels = resample_3d_data(train_data, train_labels, resampler)
            print(f"Training data successfully resampled. New shape is {train_datasource.data.shape}.")

        if scaler is not None:
            print("Scaling data...")
            train_datasource.data = scale_time_series_data_train(train_data=train_datasource.data, scaler=scaler)
            test_datasource.data = scale_time_series_data_test(test_data=test_datasource.data, trained_scaler=scaler)
            print("Data successfully scaled.")

        return train_datasource, test_datasource, train_labels, test_labels
