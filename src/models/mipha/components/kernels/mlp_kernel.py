import keras
from keras import layers, models
from models.mipha.components.kernels.data_processing_kernel import DataProcessingKernel


class MlpKernel(DataProcessingKernel):
    def __init__(self,
                 input_shape,
                 num_classes=2,
                 hidden_layers_params=None,
                 loss=None,
                 optimizer='adam',
                 metrics=None,
                 component_name=None,
                 imputer=None,
                 resampler=None,
                 scaler=None):
        """
        Initializes the MLPKernel with configurable MLP architecture. This Kernel is suitable for 2D data (n_samples, n_features).

        :param input_shape: Shape of the input features. Must be a 1D tuple.
        :type input_shape: tuple
        :param num_classes: Number of output classes for classification. Defaults to 2 (binary classification).
        :type num_classes: int
        :param hidden_layers_params: List of dictionaries containing parameters for each hidden layer.
                                     Defaults to [{'units': 64, 'activation': 'relu', 'dropout_rate':0.2}] for all layers.
        :type hidden_layers_params: None or list of dicts
        :param loss: Loss function for model compilation. Defaults to binary or sparse categorical depending on num_classes.
        :type loss: str or keras.losses.Loss
        :param optimizer: Optimizer for model compilation. Defaults to 'adam'.
        :type optimizer: str or keras.optimizers.Optimizer
        :param metrics: Metrics for model evaluation. Defaults to ['accuracy'].
        :type metrics: list of str or keras.metrics.Metric
        :param component_name: Name of this kernel component.
        :type component_name: None or str
        :param imputer: Imputer strategy to handle missing values.
        :param resampler: Resampling technique to handle class imbalance.
        :param scaler: Scaler for feature normalization.
        """
        # Input shape validation
        if len(input_shape) != 1:
            raise ValueError("Input shape must be a 1D tuple (number of features). For example, (n_features,).")

        super().__init__(component_name, imputer, resampler, scaler)

        # Build the MLP model
        self.model = build_model(
            input_shape=input_shape,
            num_classes=num_classes,
            hidden_layers_params=hidden_layers_params,
            loss=loss,
            optimizer=optimizer,
            metrics=metrics
        )

    def fit(self, x_train, y_train, *args, **kwargs):
        """
        Fits the model to the training data.

        :param x_train: Training features.
        :param y_train: Training labels.
        :param args: Additional arguments for fitting.
        :param kwargs: Additional keyword arguments for fitting.
        """
        # Process training data with imputer, resampler, scaler if needed
        _x_train, _y_train = super().process_train_data(x_train, y_train)
        self.model.fit(_x_train, _y_train, *args, **kwargs)

    def predict(self, x_test, *args, **kwargs):
        """
        Predicts the class labels for the test data.

        :param x_test: Test features.
        :param args: Additional arguments for prediction.
        :param kwargs: Additional keyword arguments for prediction.
        :return: Predicted labels for x_test.
        """
        _x_test = super().process_test_data(x_test)
        return self.model.predict(_x_test, *args, **kwargs)


def build_model(input_shape,
                num_classes=2,
                hidden_layers_params=None,
                loss=None,
                optimizer='adam',
                metrics=None):
    """
    Build an MLP model with configurable hidden layer parameters.

    :param input_shape: Shape of the input features.
    :type input_shape: tuple
    :param num_classes: Number of output classes for classification. Defaults to 2 (binary classification).
    :type num_classes: int
    :param hidden_layers_params: List of dictionaries containing parameters for each hidden layer.
                                 Each dictionary can contain 'units', 'activation', 'dropout_rate', etc.
                                 Defaults to [{'units': 64, 'activation': 'relu'}] for all layers.
    :type hidden_layers_params: list of dicts
    :param loss: Loss function for model compilation. Defaults to binary or sparse categorical depending on num_classes.
    :type loss: str or keras.losses.Loss
    :param optimizer: Optimizer for model compilation. Defaults to 'adam'.
    :type optimizer: str or keras.optimizers.Optimizer
    :param metrics: Metrics for model evaluation. Defaults to ['accuracy'].
    :type metrics: list of str or keras.metrics.Metric
    :return: Compiled MLP model.
    :rtype: keras.Model
    """

    # Default hidden layer configuration if none provided
    if hidden_layers_params is None:
        hidden_layers_params = [{'units': 64, 'activation': 'relu', 'dropout_rate': 0.2}]

    inputs = layers.Input(shape=input_shape)
    x = inputs

    # Add hidden layers according to the provided parameters
    for i, layer_params in enumerate(hidden_layers_params):
        units = layer_params.get('units', 64)
        activation = layer_params.get('activation', 'relu')
        dropout_rate = layer_params.get('dropout_rate', None)

        x = layers.Dense(units, activation=activation, name=f'hidden_layer_{i + 1}')(x)

        # Apply dropout if specified
        if dropout_rate is not None:
            x = layers.Dropout(dropout_rate, name=f'dropout_{i + 1}')(x)

    # Output layer for classification
    if num_classes == 2:
        outputs = layers.Dense(1, activation='sigmoid')(x)  # Binary classification
        default_loss = 'binary_crossentropy'
    else:
        outputs = layers.Dense(num_classes, activation='softmax')(x)  # Multi-class classification
        default_loss = 'sparse_categorical_crossentropy'

    # Create model
    model = models.Model(inputs, outputs)

    # Compile the model
    if loss is None:
        loss = default_loss
    if metrics is None:
        metrics = ['accuracy']

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.summary()

    return model
