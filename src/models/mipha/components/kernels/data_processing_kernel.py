from abc import abstractmethod

import numpy as np
from mipha.framework import MachineLearningModel

from models.mipha.utils.data_processing import scale_time_series_data_train, scale_time_series_data_test, impute_data, \
    resample_3d_data


class DataProcessingKernel(MachineLearningModel):
    """
    Utility superclass for MIPHA kernels (machine learning models) that require data pre-processing.
    Can handle both time series (3D) and 2D data.
    Subclasses can use the process_train_data and process_test_data methods to handle imputation, resampling and scaling.
    """

    def __init__(self, component_name=None, imputer=None, resampler=None, scaler=None):
        super().__init__(component_name=component_name)
        self._x_train = None
        self._y_train = None
        self._x_test = None

        self.imputer = imputer
        self.resampler = resampler
        self.scaler = scaler

    @abstractmethod
    def fit(self, x_train, y_train, *args, **kwargs):
        return NotImplemented

    @abstractmethod
    def predict(self, x_test, *args, **kwargs):
        return NotImplemented

    def process_train_data(self, x_train, y_train):
        """
        Processes the training data by applying imputation, resampling, and scaling
        as specified by the model's parameters.

        :param x_train: The input features for training.
        :type x_train: ndarray or DataFrame
        :param y_train: The target labels for training.
        :type y_train: ndarray or Series

        :return: A tuple containing the processed features (_x_train)
                 and the processed labels (_y_train) after applying
                 imputation, resampling, and scaling based on the dimensionality
                 of the data (2D or 3D).
        :rtype: tuple
        """
        print(f"[{self.component_name}] Processing train data... (shapes: x {x_train.shape}, y {y_train.shape})")

        # Copy input data to avoid modifying original
        _x_train = x_train.copy()
        _y_train = y_train.copy()

        # Function to handle 2D data processing
        def process_2d_train_data(x, y):
            x = self.imputer.fit_transform(x) if self.imputer is not None else x
            x, y = self.resampler.fit_resample(x, y) if self.resampler is not None else (x, y)
            x = self.scaler.fit_transform(x) if self.scaler is not None else x
            print("")
            return x, y

        # Function to handle 3D data processing
        def process_3d_train_data(x, y):
            x = impute_data(x, self.imputer) if self.imputer is not None else x
            x, y = resample_3d_data(x, y, self.resampler) if self.resampler is not None else (x, y)
            x = scale_time_series_data_train(x, self.scaler) if self.scaler is not None else x
            return x, y

        # Choose processing path based on data dimensionality
        dimension = len(_x_train.shape)
        if dimension == 2:
            _x_train, _y_train = process_2d_train_data(_x_train, _y_train)
        elif dimension == 3:
            _x_train, _y_train = process_3d_train_data(_x_train, _y_train)
        else:
            AttributeError(f'Unsupported dimension: {dimension}')

        self._x_train = _x_train
        self._y_train = _y_train

        print(f"[{self.component_name}] Train data processed (shapes: x {_x_train.shape}, y {_y_train.shape})")
        return _x_train, _y_train

    def process_test_data(self, x_test):
        """
        Processes the test data by applying imputation and scaling
        based on the parameters learned during training.

        :param x_test: The input features for testing.
        :type x_test: ndarray or DataFrame

        :return: The processed features (_x_test) after
                 applying imputation and scaling as needed.
        :rtype: ndarray or DataFrame
        """

        print(f"[{self.component_name}] Processing test data... (shape: x {x_test.shape})")
        # Copy input data to avoid modifying original
        _x_test = x_test.copy()

        # Function to handle 2D data processing
        def process_2d_test_data(x):
            x = self.imputer.transform(x) if self.imputer is not None else x
            x = self.scaler.transform(x) if self.scaler is not None else x
            return x

        # Function to handle 3D data processing
        def process_3d_test_data(x):
            x = impute_data(x, self.imputer) if self.imputer is not None else x
            x = scale_time_series_data_test(x, self.scaler) if self.scaler is not None else x
            return x

        # Choose processing path based on data dimensionality
        dimension = len(_x_test.shape)
        if dimension == 2:
            _x_test = process_2d_test_data(_x_test)
        elif dimension == 3:
            _x_test = process_3d_test_data(_x_test)
        else:
            AttributeError(f'Unsupported dimension: {dimension}')

        self._x_test = _x_test

        print(f"[{self.component_name}]Test data processed (shape: x {_x_test.shape})")
        return _x_test
