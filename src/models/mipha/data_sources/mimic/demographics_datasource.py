import warnings

import numpy as np
import pandas as pd
from mipha.framework import DataSource

from models.mipha.data_sources.mimic.record_to_matrix_conversion import create_learning_matrix, create_mask_from_records


class DemographicsDataSource(DataSource):
    def __init__(self, data_type, name, data, mask=None):
        """
        Represents a data source with a specific type, name, and data.
        This data source is suitable for demographics data extracted as dictionaries where keys are patient IDs,
        and values are sub-dictionaries. The keys of the sub-dictionaries are timestamps (observation dates)
        and their values are dictionaries containing the data (age and gender).

        :param data_type: The type of data contained in the data source.
        :type data_type: str

        :param name: The name of the data source.
        :type name: str

        :param data: The actual data of the data source. Refer to the description above for the format.

        :param mask: The mask associated with the data source, given as a list of (patient_id, timestamp) tuples.
                     When given, the numpy array associated with the `data` parameter will match this mask.
                     If not, a mask will be created from the data directly if the data is provided as a dictionary (formatted as above).
        """

        super().__init__(data_type=data_type, name=name, data=data)

        self.mask = mask if mask is not None else create_mask_from_records(patient_records=data)

        demographics_records = {
            patient_id: {
                timestamp: pd.DataFrame([data])
                for timestamp, data in patient_data.items()
            }
            for patient_id, patient_data in data.items()
        }

        self.data = create_learning_matrix(patient_records=demographics_records, mask=self.mask)

    def extend_timesteps(self, n_timesteps):
        """
        Utility function to reshape the data to a 3D array of shape (n_samples, n_timesteps, n_features).
        Features are simply repeated for each timestep.

        :param n_timesteps: Number of timesteps to extend the data to.
        :type n_timesteps: int
        """
        # For 2D data, repeat the features along the time axis to create a 3D array.
        if self.data.ndim == 2:
            n_samples, n_features = self.data.shape
            self.data = np.repeat(self.data[:, np.newaxis, :], repeats=n_timesteps, axis=1)
            print(f"Data reshaped from {n_samples, n_features} to {self.data.shape}")

        # For 3D data with a single timestep, repeat along the timestep axis.
        elif self.data.ndim == 3 and self.data.shape[1] == 1:
            n_samples, _, n_features = self.data.shape
            self.data = np.repeat(self.data, repeats=n_timesteps, axis=1)
            print(f"Data extended from {n_samples, 1, n_features} to {self.data.shape}")

        else:
            warnings.warn(f"Data shape {self.data.shape} does not require reshaping.")

        return self.data

    def flatten_data(self):
        """
        Utility function to reshape the data to a 2D array. The array is returned and set as the `self.data` attribute.
        """
        if self.data.ndim > 2:
            self.data = np.reshape(self.data, newshape=(self.data.shape[0], self.data.shape[-1]))
        print(f"Data was reshaped into an array of shape {self.data.shape}")
        return self.data
