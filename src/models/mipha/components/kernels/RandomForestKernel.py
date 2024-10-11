from mipha.framework import MachineLearningModel
from sklearn.ensemble import RandomForestClassifier


class RandomForestKernel(MachineLearningModel):
    def __init__(self, component_name, **kwargs):
        """
        Initializes the RandomForestKernel.

        :param component_name: Name of the component.
        :param kwargs: Additional keyword arguments passed to RandomForestClassifier.
                       Refer to scikit-learn's RandomForestClassifier for a list of all available parameters.
        """
        super().__init__(component_name=component_name)
        self.model = RandomForestClassifier(**kwargs)

    def fit(self, x_train, y_train, *args, **kwargs):
        """
        Fits the Random Forest model to the training data.

        :param x_train: Training features.
        :param y_train: Training labels.
        :param args: Additional arguments.
        :param kwargs: Additional keyword arguments.
        """
        self.model.fit(x_train, y_train, *args, **kwargs)

    def predict(self, x_test, *args, **kwargs):
        """
        Predicts the class labels for the test data.

        :param x_test: Test features.
        :return: Predicted labels for x_test.
        """
        return self.model.predict(x_test)

    def predict_proba(self, x_test, *args, **kwargs):
        """
        Predicts class probabilities for the test data.

        :param x_test: Test features.
        :return: Predicted class probabilities for x_test.
        """
        return self.model.predict_proba(x_test)
