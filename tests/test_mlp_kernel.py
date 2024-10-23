import numpy as np
import pytest

from models.mipha.components.kernels.mlp_kernel import MlpKernel


@pytest.fixture
def setup_mlp_kernel():
    input_shape = (10,)  # 10 features
    num_classes = 2
    mlp_kernel = MlpKernel(
        input_shape=input_shape,
        num_classes=num_classes,
        hidden_layers_params=None,  # No params provided
        loss='binary_crossentropy',
        optimizer='adam'
    )
    return mlp_kernel


def test_default_hidden_layers_params(setup_mlp_kernel):
    mlp_kernel = setup_mlp_kernel
    # Check that the model has been built with default parameters
    model = mlp_kernel.model
    assert len(model.layers) == 4 # 1 input + 1 hidden + 1 dropout + 1 output layer
    assert model.layers[1].units == 64  # Default units for the hidden layer
    assert model.layers[1].activation.__name__ == 'relu'  # Default activation


def test_custom_hidden_layers_params():
    input_shape = (10,)
    num_classes = 2
    hidden_layers_params = [
        {'units': 64, 'activation': 'relu'},
        {'units': 32, 'activation': 'relu'},
    ]
    mlp_kernel = MlpKernel(
        input_shape=input_shape,
        num_classes=num_classes,
        hidden_layers_params=hidden_layers_params,
        loss='binary_crossentropy',
        optimizer='adam'
    )

    model = mlp_kernel.model
    assert len(model.layers) == 4  # 1 input + 2 hidden + 1 output layer
    assert model.layers[1].units == 64  # First layer units
    assert model.layers[2].units == 32  # Second layer units


def test_output_shape_binary_classification():
    input_shape = (10,)  # 10 features
    num_classes = 2
    mlp_kernel = MlpKernel(
        input_shape=input_shape,
        num_classes=num_classes,
        hidden_layers_params=None,
        loss='binary_crossentropy',
        optimizer='adam'
    )
    x_train = np.random.rand(100, 10)  # 100 samples, 10 features
    y_train = np.random.randint(0, 2, size=(100,))  # Binary labels
    mlp_kernel.fit(x_train, y_train, epochs=1, batch_size=16)

    predictions = mlp_kernel.model.predict(np.random.rand(10, 10))  # 10 samples for prediction
    assert predictions.shape == (10, 1)  # Output should be of dimension 1


def test_output_shape_multi_classification():
    input_shape = (10,)  # 10 features
    num_classes = 3  # More than two classes
    mlp_kernel = MlpKernel(
        input_shape=input_shape,
        num_classes=num_classes,
        hidden_layers_params=None,
        loss='sparse_categorical_crossentropy',
        optimizer='adam'
    )
    x_train = np.random.rand(100, 10)  # 100 samples, 10 features
    y_train = np.random.randint(0, 3, size=(100,))  # Multi-class labels
    mlp_kernel.fit(x_train, y_train, epochs=1, batch_size=16)

    predictions = mlp_kernel.model.predict(np.random.rand(10, 10))  # 10 samples for prediction
    assert predictions.shape == (10, 3)  # Output should be of dimension equal to number of classes


def test_invalid_input_shape():
    with pytest.raises(ValueError):
        MlpKernel(input_shape=(10, 10))  # Invalid input shape for MLP
