import copy

import numpy as np
from mipha.framework import FeatureExtractor
from keras import layers, models


class CnnAutoencoderFeatureExtractor(FeatureExtractor):
    def __init__(self, component_name,
                 input_shape,
                 latent_dim,
                 n_layers=2,
                 n_filters=64,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 loss='mse',
                 optimizer='adam',
                 managed_data_types: list[str] = None):
        """
        Create a customizable autoencoder-based feature extractor for time-series feature extraction.
        The autoencoder employs 1D convolution layers.

        :param component_name: Name of the component
        :type component_name: str
        :param managed_data_types:  A list of data source type names to extract features from. Whenever feature extraction needs to be performed for a DataSource with attribute dataType equal to managed_data_types, the MiphaPredictor: will feed it to this FeatureExtractor.
        :type managed_data_types: list[str]
        :param input_shape: Shape of the input (n_timesteps, n_features).
        :type input_shape: tuple
        :param latent_dim: Dimensionality of the latent (encoded) space.
        :type latent_dim: int
        :param n_layers: Number of Conv1D layers in the encoder. Default is 2.
        :type n_layers: int
        :param n_filters: Number of filters for the first Conv1D layer (doubles with depth). Default is 64.
        :type n_filters: int
        :param kernel_size: Size of the convolutional kernel. Default is 3.
        :type kernel_size: int
        :param strides: Strides of the convolutional kernel. Default is 1.
        :type strides: int
        :param activation: Activation function for layers. Default is 'relu'.
        :type activation: str
        :param loss: Loss function to optimize. Default is 'mse'.
        :type loss: str
        :param optimizer: Optimizer to use during training. Default is 'adam'.
        """

        super().__init__(component_name=component_name, managed_data_types=managed_data_types)
        self.autoencoder, self.encoder, self.decoder = create_autoencoder(
            input_shape=input_shape,
            latent_dim=latent_dim,
            n_layers=n_layers,
            n_filters=n_filters,
            kernel_size=kernel_size,
            strides=strides,
            activation=activation,
            loss=loss,
            optimizer=optimizer,
        )

    def fit(self, x, *args, **kwargs):
        return self.autoencoder.fit(x=x, y=x, *args, **kwargs)  # in an autoencoder the target is the input

    def extract_features(self, x):
        encoder_input = copy.deepcopy(x)

        if len(x.shape) == 2:
            encoder_input = np.expand_dims(x, axis=0)
        elif len(x.shape) != 3:
            raise ValueError(f"Input must be 2D (single observation) or 3D (batch input). Found shape {x.shape}.")

        features = self.encoder.predict(encoder_input)
        return features


def create_autoencoder(
        input_shape,
        latent_dim,
        n_layers=2,
        n_filters=64,
        kernel_size=3,
        strides=1,
        activation='relu',
        loss='mse',
        optimizer='adam'
):
    """
    Create a customizable autoencoder for time-series feature extraction.
        :param input_shape: Shape of the input (n_timesteps, n_features).
        :type input_shape: tuple
        :param latent_dim: Dimensionality of the latent (encoded) space.
        :type latent_dim: int
        :param n_layers: Number of Conv1D layers in the encoder. Default is 2.
        :type n_layers: int
        :param n_filters: Number of filters for the first Conv1D layer (doubles with depth). Default is 64.
        :type n_filters: int
        :param kernel_size: Size of the convolutional kernel. Default is 3.
        :type kernel_size: int
        :param strides: Strides of the convolutional kernel. Default is 1.
        :type strides: int
        :param activation: Activation function for layers. Default is 'relu'.
        :type activation: str
        :param loss: Loss function to optimize. Default is 'mse'.
        :type loss: str
        :param optimizer: Optimizer to use during training. Default is 'adam'.

        :return: A tuple containing:
            - autoencoder: Complete autoencoder model.
            - encoder: Encoder part for feature extraction.
            - decoder: Decoder model.
        :rtype: tuple
    """

    # Create encoder and decoder models
    encoder_input = layers.Input(shape=input_shape)
    encoder = make_encoder(activation, encoder_input, kernel_size, latent_dim, n_filters, n_layers, strides)
    decoder = make_decoder(activation, input_shape, kernel_size, latent_dim, n_filters, n_layers, strides)

    # Create and compile autoencoder
    autoencoder_input = encoder_input
    encoded = encoder(autoencoder_input)
    decoded = decoder(encoded)
    autoencoder = models.Model(autoencoder_input, decoded, name="autoencoder")
    autoencoder.compile(loss=loss, optimizer=optimizer)

    return autoencoder, encoder, decoder


def make_encoder(activation, encoder_input, kernel_size, latent_dim, base_n_filters, n_layers, strides):
    x = encoder_input

    # Convolution / MaxPool blocks
    for i in range(n_layers):
        x = layers.Conv1D(
            # the number of filters increases exponentially as the network goes deeper to capture more complex features generated each layer
            filters=base_n_filters * (2 ** i),
            kernel_size=kernel_size,
            strides=strides,
            activation=activation,
            padding='same')(x)  # "same" padding maintains the dimension
        x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Flatten()(x)
    latent = layers.Dense(latent_dim, activation=activation)(x)
    encoder = models.Model(encoder_input, latent, name="encoder")
    return encoder


def make_decoder(activation, input_shape, kernel_size, latent_dim, base_n_filters, n_layers, strides):
    decoder_input = layers.Input(shape=(latent_dim,))
    n_timesteps, n_features = input_shape  # initial number of timesteps
    downsampling_factor = 2 ** n_layers  # each MaxPool1D divides the number of timesteps by 2
    reduced_n_timesteps = n_timesteps // downsampling_factor  # number of timesteps after reduction
    n_filters = base_n_filters * (2 ** (n_layers - 1))  # number of filters in the last encoder layer

    # matches the dimension of the latent layer from the encoder
    x = layers.Dense(units=reduced_n_timesteps * n_filters)(decoder_input)

    # converts the flat output back to a shape processable by the next layers
    x = layers.Reshape((reduced_n_timesteps, n_filters))(x)

    # "reverse" loop to reconstruct the input
    for i in range(n_layers - 1, -1, -1):  # 1 less layer because the last one is created manually
        x = layers.Conv1DTranspose(
            filters=base_n_filters * (2 ** i),
            kernel_size=kernel_size,
            strides=strides,
            activation=activation,
            padding='same')(x)
        x = layers.UpSampling1D(size=2)(x)

    decoder_output = layers.Conv1DTranspose(
        filters=n_features,
        kernel_size=kernel_size,
        strides=strides,
        activation='linear',
        padding='same')(x)

    decoder = models.Model(decoder_input, decoder_output, name="decoder")
    return decoder
