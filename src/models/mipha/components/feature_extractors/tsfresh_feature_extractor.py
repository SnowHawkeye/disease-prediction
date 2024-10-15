import copy
import warnings

import pandas as pd
from mipha.framework import FeatureExtractor
from tsfresh import extract_features, select_features
from tsfresh.feature_extraction import ComprehensiveFCParameters
from tsfresh.utilities.dataframe_functions import impute


class TsfreshFeatureExtractor(FeatureExtractor):
    def __init__(self,
                 component_name,
                 managed_data_types: list[str] = None,
                 default_fc_parameters: dict = ComprehensiveFCParameters()
                 ):
        """
        Create a customizable autoencoder-based feature extractor for time-series feature extraction.
        The autoencoder employs 1D convolution layers.

        :param component_name: Name of the component
        :type component_name: str
        :param managed_data_types:  A list of data source type names to extract features from. Whenever feature extraction needs to be performed for a DataSource with attribute dataType equal to managed_data_types, the MiphaPredictor: will feed it to this FeatureExtractor.
        :type managed_data_types: list[str]
        :param default_fc_parameters:  The default parameters to use for feature extraction, passed to tsfresh.
        """
        super().__init__(component_name=component_name, managed_data_types=managed_data_types)
        self.learned_best_features = None
        self.default_fc_parameters = default_fc_parameters

    def extract_features(self, x):
        x_dataframe = copy.deepcopy(x)
        if len(x.shape) == 2:
            x_dataframe = [x]
        elif len(x.shape) != 3:
            raise ValueError(f"Input must be 2D (single observation) or 3D (batch input). Found shape {x.shape}.")

        if self.learned_best_features is None:
            warnings.warn(
                "No learned best features found. All computable features will be returned. Call `fit` first for a filtered selection of features."
            )

        x_dataframe = self.format_data_tsfresh(x_dataframe)
        extracted_features = extract_features(  # this step seems to take a long time with default parameters
            x_dataframe,
            column_id='id',
            default_fc_parameters=self.default_fc_parameters
        )
        impute(extracted_features)

        if self.learned_best_features is not None:
            print("Learned best features found. Feature extraction will return a filtered selection of features.")
            extracted_features = extracted_features[self.learned_best_features]

        return extracted_features.to_numpy()

    def fit(self, x, y):
        x_dataframe = self.format_data_tsfresh(x)
        extracted_features = extract_features(
            x_dataframe,
            column_id='id',
            default_fc_parameters=self.default_fc_parameters
        )
        impute(extracted_features)
        features_filtered = select_features(extracted_features, pd.Series(y))
        self.learned_best_features = features_filtered.columns
        return features_filtered.to_numpy()

    def format_data_tsfresh(self, x):
        # Formatting data for tsfresh
        x_dataframe = [pd.DataFrame(table) for table in x]  # tsfresh requires DataFrames
        for i, df in enumerate(x_dataframe):
            df["id"] = i  # the id identifies the matrices when the DataFrames are stacked
        x_dataframe = pd.concat(x_dataframe)
        return x_dataframe
