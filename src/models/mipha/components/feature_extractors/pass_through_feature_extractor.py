from mipha.framework import FeatureExtractor


class PassThroughFeatureExtractor(FeatureExtractor):
    def __init__(self, component_name, managed_data_types: list[str] = None):
        """
        This feature extractor returns the data as-is.
        """
        super().__init__(component_name=component_name, managed_data_types=managed_data_types)

    def extract_features(self, x):
        return x
