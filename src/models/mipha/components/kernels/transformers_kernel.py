"""
This implementation is based on the example provided by the Keras team:
https://keras.io/examples/timeseries/timeseries_classification_transformer/

Modifications have been made to adapt the model to the specific needs of this project.
"""

from models.mipha.components.kernels.data_processing_kernel import DataProcessingKernel
from keras import layers, Input, Model


class TransformersKernel(DataProcessingKernel):
    def __init__(self,
                 input_shape,
                 num_classes=2,
                 head_size=64,
                 num_heads=4,
                 ff_dim=128,
                 num_transformer_blocks=4,
                 mlp_units=None,
                 dropout=0.0,
                 mlp_dropout=0.0,
                 loss=None,
                 optimizer='adam',
                 metrics=None,
                 component_name=None,
                 imputer=None,
                 resampler=None,
                 scaler=None,
                 ):
        """
        Transformer-based implementation of MIPHA MachineLearningModel. Optionally imputes, resamples and scales the data.

        :param input_shape: Shape of the input data (timesteps, features).
        :param num_classes: Number of output classes for classification.
        :param head_size: Size of the attention head.
        :param num_heads: Number of attention heads.
        :param ff_dim: Number of filters in the feed-forward network inside the transformer.
        :param num_transformer_blocks: Number of transformer blocks.
        :param mlp_units: List of hidden layer sizes in the MLP classifier. Defaults to one layer of size 128.
        :param dropout: Dropout rate within the transformer blocks.
        :param mlp_dropout: Dropout rate for the MLP classifier.
        :param loss: Loss function to use. If None, will default to binary or sparse categorical based on num_classes.
        :param optimizer: Optimizer to be used for model compilation. Defaults to Adam optimizer with a learning rate of 1e-4.
        :param metrics: List of metrics to be evaluated by the model during training and testing.
        :param component_name: Name of the component.
        :param imputer: Data imputer to handle missing values.
        :param resampler: Data resampler for adjusting the dataset.
        :param scaler: Data scaler for feature normalization.
        """

        super().__init__(component_name=component_name, imputer=imputer, resampler=resampler, scaler=scaler)

        # Default values to avoid mutable arguments
        if mlp_units is None:
            mlp_units = [128]
        if metrics is None:
            metrics = ["sparse_categorical_accuracy"]

        self.model = build_model(
            input_shape=input_shape,
            num_classes=num_classes,
            head_size=head_size,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_transformer_blocks=num_transformer_blocks,
            mlp_units=mlp_units,
            dropout=dropout,
            mlp_dropout=mlp_dropout,
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


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.0):
    """
    Builds a single Transformer encoder block.

    :param inputs: Input tensor.
    :param head_size: Size of each attention head.
    :param num_heads: Number of attention heads.
    :param ff_dim: Number of filters in the feed-forward network.
    :param dropout: Dropout rate for the layers.
    :return: Output tensor after the Transformer encoder block.
    """
    # Attention and Normalization
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs  # residual connection (adding input to output to mitigate the vanishing gradient problem)

    # Feed Forward Part
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res  # residual connection (see above)


def build_model(input_shape,
                num_classes,
                head_size,
                num_heads,
                ff_dim,
                num_transformer_blocks,
                mlp_units,
                dropout=0.0,
                mlp_dropout=0.0,
                loss=None,
                optimizer='adam',
                metrics=None
                ):
    """
    Builds a Transformer-based model for classification.

    :param input_shape: Shape of the input data (timesteps, features).
    :param num_classes: Number of output classes for classification.
    :param head_size: Size of each attention head.
    :param num_heads: Number of attention heads.
    :param ff_dim: Number of filters in the feed-forward network inside the transformer.
    :param num_transformer_blocks: Number of transformer blocks.
    :param mlp_units: List of hidden layer sizes in the MLP classifier.
    :param dropout: Dropout rate within the transformer blocks.
    :param mlp_dropout: Dropout rate for the MLP classifier.
    :param loss: Loss function to use. If None, will default to binary or sparse categorical based on num_classes.
    :param optimizer: Optimizer to be used for model compilation. Defaults to Adam optimizer with a learning rate of 1e-4.
    :param metrics: List of metrics to be evaluated by the model during training and testing.
    :return: Keras Model instance.
    """
    inputs = Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)

    # Output layer: adapt for multi-class classification
    if num_classes == 2:
        outputs = layers.Dense(1, activation='sigmoid')(x)  # Binary classification
        default_loss = 'binary_crossentropy'
    else:
        outputs = layers.Dense(num_classes, activation='softmax')(x)  # Multi-class classification
        default_loss = 'sparse_categorical_crossentropy'

    # Default loss, metrics, and optimizer
    if loss is None:
        loss = default_loss
    if metrics is None:
        metrics = ['accuracy']

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=metrics,
    )
    model.summary()
    return model
