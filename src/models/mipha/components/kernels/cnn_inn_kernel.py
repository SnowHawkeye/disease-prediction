import keras

from models.mipha.components.kernels.data_processing_kernel import DataProcessingKernel
from keras import layers, models

from models.mipha.utils.layers.involution import Involution


class CnnInnKernel(DataProcessingKernel):

    def __init__(self,
                 input_shape,
                 num_classes=2,
                 num_convolution_layers=1,
                 num_involution_layers=1,
                 convolution_params=None,
                 involution_params=None,
                 loss=None,
                 optimizer='adam',
                 metrics=None,
                 component_name=None,
                 imputer=None,
                 resampler=None,
                 scaler=None):
        super().__init__(component_name, imputer, resampler, scaler)

        self.model = build_model(
            input_shape,
            num_classes=num_classes,
            num_convolution_layers=num_convolution_layers,
            num_involution_layers=num_involution_layers,
            convolution_params=convolution_params,
            involution_params=involution_params,
            loss=loss,
            optimizer=optimizer,
            metrics=metrics,
        )

    def fit(self, x_train, y_train, *args, **kwargs):
        """
        Fits the model to the training data.

        :param x_train: Training features.
        :param y_train: Training labels.
        :param args: Additional arguments for fitting.
        :param kwargs: Additional keyword arguments for fitting.
        """
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
                num_convolution_layers=1,
                num_involution_layers=1,
                convolution_params=None,
                involution_params=None,
                loss=None,
                optimizer='adam',
                metrics=None
                ):
    """
    Build a neural network model with user-defined Conv1D and Involution layers.

    :param input_shape: The shape of the input data (timesteps, features).
    :type input_shape: tuple
    :param num_convolution_layers: Number of Conv1D layers to include. Defaults to 1.
    :type num_convolution_layers: int
    :param num_involution_layers: Number of Involution layers to include. Defaults to 1.
    :type num_involution_layers: int
    :param convolution_params: Hyperparameters for Conv1D layers (filters, kernel_size, strides, etc.).
    :type convolution_params: dict or None
    :param involution_params: Hyperparameters for Involution layers.
    :type involution_params: dict or None
    :param num_classes: Number of output classes for classification. Defaults to 2 (binary classification).
    :type num_classes: int
    :param loss: Loss function to use. If None, will default to binary or sparse categorical based on num_classes.
    :type loss: str or keras.losses.Loss or None
    :param optimizer: Optimizer to use for model compilation. Defaults to 'adam'.
    :type optimizer: str or keras.optimizers.Optimizer
    :param metrics: List of metrics to use for model evaluation. Defaults to ['accuracy'].
    :type metrics: list or None
    :return: The compiled neural network model.
    :rtype: keras.Model
    """

    defaults_conv_params = {
        'filters': 64,
        'kernel_size': 3,
        'strides': 1,
        'padding': 'same',
        'activation': 'relu'
    }

    defaults_inv_params = {
        'channel': 64,
        'group_number': 4,
        'kernel_size': 3,
        'stride': 1,
        'reduction_ratio': 4,
        'name': 'involution'
    }

    convolution_params = defaults_conv_params if convolution_params is None else convolution_params
    involution_params = defaults_inv_params if involution_params is None else involution_params

    inputs = layers.Input(shape=input_shape)
    x = inputs

    # Add Conv1D layers if specified
    for i in range(num_convolution_layers):
        x = layers.Conv1D(
            filters=convolution_params.get('filters', defaults_conv_params['filters']),
            kernel_size=convolution_params.get('kernel_size', defaults_conv_params['kernel_size']),
            strides=convolution_params.get('strides', defaults_conv_params['strides']),
            padding=convolution_params.get('padding', defaults_conv_params['padding']),
            activation=convolution_params.get('activation', defaults_conv_params['activation']),
            name=f'conv1d_layer_{i + 1}'
        )(x)

    # Reshape the input of Involution layers to 4D
    x = layers.Reshape((-1, 1, x.shape[-1]))(x)  # Reshape to (batch_size, height, width, channels)

    # Add Involution layers if specified
    for i in range(num_involution_layers):
        x, _ = Involution(
            channel=involution_params.get('channel', defaults_inv_params['channel']),
            group_number=involution_params.get('group_number', defaults_inv_params['group_number']),
            kernel_size=involution_params.get('kernel_size', defaults_inv_params['kernel_size']),
            stride=involution_params.get('stride', defaults_inv_params['stride']),
            reduction_ratio=involution_params.get('reduction_ratio', defaults_inv_params['reduction_ratio']),
            name=f'involution_layer_{i + 1}'
        )(x)

    x = layers.Reshape((-1, x.shape[-1]))(x)  # Reshape to (batch_size, height * width, channels)

    # Global average pooling and output layer for prediction
    x = layers.GlobalAveragePooling1D()(x)

    # Output layer: adapt for multi-class classification
    if num_classes == 2:
        outputs = layers.Dense(1, activation='sigmoid')(x)  # Binary classification
        default_loss = 'binary_crossentropy'
    else:
        outputs = layers.Dense(num_classes, activation='softmax')(x)  # Multi-class classification
        default_loss = 'sparse_categorical_crossentropy'

    # Create model
    model = models.Model(inputs, outputs)

    # Default loss, metrics, and optimizer
    if loss is None:
        loss = default_loss
    if metrics is None:
        metrics = ['accuracy']

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.summary()

    return model
