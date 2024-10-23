from models.mipha.components.kernels.data_processing_kernel import DataProcessingKernel


class SimpleDataProcessingKernel(DataProcessingKernel):
    def __init__(self, model, component_name=None, imputer=None, resampler=None, scaler=None, **kwargs):
        """
        Initializes a Kernel with the given model. The Kernel inherits from DataProcessingKernel, which means
        that if imputer and/or resampler and/or scaler are provided, they will be used to process the data.

        :param model: The class of the model.
        :param component_name: Name of the component.
        :param kwargs: Additional keyword arguments passed to the model.
        """
        super().__init__(component_name=component_name, imputer=imputer, resampler=resampler, scaler=scaler)
        self.model = model(**kwargs)

    def fit(self, x_train, y_train, *args, **kwargs):
        """
        Fits the model to the training data.

        :param x_train: Training features.
        :param y_train: Training labels.
        :param args: Additional arguments.
        :param kwargs: Additional keyword arguments.
        """

        _x_train, _y_train = super().process_train_data(x_train, y_train)
        self.model.fit(_x_train, _y_train, *args, **kwargs)

    def predict(self, x_test, *args, **kwargs):
        """
        Predicts the class labels for the test data.

        :param x_test: Test features.
        :return: Predicted labels for x_test.
        """
        _x_test = super().process_test_data(x_test)
        return self.model.predict(_x_test)

    def predict_proba(self, x_test, *args, **kwargs):
        """
        Predicts class probabilities for the test data.

        :param x_test: Test features.
        :return: Predicted class probabilities for x_test.
        """
        _x_test = super().process_test_data(x_test)
        return self.model.predict_proba(_x_test)
