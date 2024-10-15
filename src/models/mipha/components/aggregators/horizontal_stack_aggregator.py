import numpy as np
from mipha.framework import Aggregator


class HorizontalStackAggregator(Aggregator):
    """
    Simple aggregator stacking features horizontally.
    If all inputs are 3D, they are concatenated along the feature axis.
    Otherwise, all inputs are flattened to 2D.
    It is assumed that the first dimension is n_samples.
    """

    def aggregate_features(self, features):
        """
        :param features: List of features to be combined.
        :return: Aggregated features.
        """
        # Ensure that all inputs have the same number of samples
        num_samples = features[0].shape[0]
        if not all(f.shape[0] == num_samples for f in features):
            raise ValueError("All feature arrays must have the same number of samples")

        # Check if all inputs have the same dimension
        all_same_dimension = all(f.ndim == features[0].ndim for f in features)

        if all_same_dimension:
            # Concatenate all inputs along the last axis (feature axis)
            aggregated_features = np.concatenate(features, axis=-1)
        else:
            # Flatten each feature array to 2D if it has more than 2 dimensions and stack along the feature axis
            flattened_features = [f.reshape(f.shape[0], -1) if f.ndim > 2 else f for f in features]
            aggregated_features = np.hstack(flattened_features)

        return aggregated_features
