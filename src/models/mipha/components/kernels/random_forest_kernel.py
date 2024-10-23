from sklearn.ensemble import RandomForestClassifier

from models.mipha.components.kernels.simple_data_processing_kernel import SimpleDataProcessingKernel


class RandomForestKernel(SimpleDataProcessingKernel):
    def __init__(self, component_name=None, imputer=None, resampler=None, scaler=None, **kwargs):
        """
        Initializes the RandomForestKernel.

        ⚠️This class was kept to ensure compatibility with existing experiments.
        The simpler approach is to use SimpleDataProcessingKernel class instead, with model=RandomForestClassifier.

        :param component_name: Name of the component.
        :param kwargs: Additional keyword arguments passed to RandomForestClassifier.
                       Refer to scikit-learn's RandomForestClassifier for a list of all available parameters.
        """
        super().__init__(
            model=RandomForestClassifier,
            component_name=component_name,
            imputer=imputer,
            resampler=resampler,
            scaler=scaler
        )
