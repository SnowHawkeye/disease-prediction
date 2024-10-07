import warnings

from mipha.framework import DataSource
from models.mipha.data_sources.mimic.record_to_matrix_conversion import create_learning_matrix, create_mask_from_records
from models.mipha.utils.data_processing import impute_data, scale_time_series_data_train, scale_time_series_data_test
import numpy as np
from models.mipha.utils.data_processing import mask_train_test_split


class PatientRecordDatasource(DataSource):
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

        # Only call create_learning_matrix if data is not already a numpy array
        if isinstance(data, dict):
            self.mask = mask if mask is not None else create_mask_from_records(patient_records=data)
            self.data = create_learning_matrix(patient_records=data, mask=self.mask)
        else:
            self.data = data

        if mask is None:
            warnings.warn(
                "Mask could not be created from the data and was set to None. In particular, the split_train_test method will not work."
            )

        self.mask = mask

    def impute_data(self, imputer):
        """
        Impute this data source's data using the provided imputer.
        """
        self.data = impute_data(data=self.data, imputer=imputer)

    def split_train_test(self, test_size=0.2, random_seed=None, scaler=None):
        """
        Split the data source's data into two new data sources: one for training and one for testing.
        Optionally scales the data.

        :param test_size: Proportion of patient IDs to include in the test split (default is 0.2).
        :type test_size: float

        :param random_seed: Seed for the random number generator for reproducibility (default is None).
        :type random_seed: int or None

        :param scaler: The scaler used to scale the data. Defaults to None. If no scaler is provided, the data isn't scaled.

        :return: Two new PatientRecordDatasource instances for training and testing.
        :rtype: tuple(PatientRecordDatasource, PatientRecordDatasource)
        """

        # Split the mask into training and testing masks
        train_mask, test_mask = mask_train_test_split(self.mask, test_size=test_size, random_seed=random_seed)

        # Split the data according to the new masks
        train_data, test_data = [], []
        for (patient_id, timestamp), record in zip(self.mask, self.data):
            if (patient_id, timestamp) in train_mask:
                train_data.append(record)
            else:
                test_data.append(record)

        # Convert lists back into numpy arrays
        train_data = np.array(train_data)
        test_data = np.array(test_data)

        # Create new data sources for training and testing
        train_datasource = PatientRecordDatasource(data_type=self.data_type,
                                                   name=f"{self.name}_train",
                                                   data=train_data,
                                                   mask=train_mask)

        test_datasource = PatientRecordDatasource(data_type=self.data_type,
                                                  name=f"{self.name}_test",
                                                  data=test_data,
                                                  mask=test_mask)

        if scaler is not None:
            train_datasource.data = scale_time_series_data_train(train_data=train_datasource.data, scaler=scaler)
            test_datasource.data = scale_time_series_data_test(test_data=test_datasource.data, trained_scaler=scaler)

        return train_datasource, test_datasource
