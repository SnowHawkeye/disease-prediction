import numpy as np
from keras import Model, Input

from models.mipha.components.kernels.transformers_kernel import build_model, transformer_encoder, TransformersKernel


def test_build_model():
    input_shape = (24, 20)  # Example input shape
    n_classes = 3
    model = build_model(input_shape, n_classes, head_size=64, num_heads=4, ff_dim=128, num_transformer_blocks=2,
                        mlp_units=[64, 32], dropout=0.1, mlp_dropout=0.1)

    assert isinstance(model, Model), "build_model should return a Keras Model"
    assert model.input_shape == (None, 24, 20), f"Expected input shape (None, 24, 20) but got {model.input_shape}"
    assert model.output_shape == (None, n_classes), f"Expected output shape (None, 3) but got {model.output_shape}"


def test_build_model_binary():
    input_shape = (24, 20)  # Example input shape
    n_classes = 2
    model = build_model(input_shape, n_classes, head_size=64, num_heads=4, ff_dim=128, num_transformer_blocks=2,
                        mlp_units=[64, 32], dropout=0.1, mlp_dropout=0.1)

    assert isinstance(model, Model), "build_model should return a Keras Model"
    assert model.input_shape == (None, 24, 20), f"Expected input shape (None, 24, 20) but got {model.input_shape}"
    assert model.output_shape == (None, 1), f"Expected output shape (None, 1) but got {model.output_shape}"


def test_transformer_encoder():
    input_shape = (10, 20)  # Example input with 10 timesteps and 20 features
    inputs = Input(shape=input_shape)
    outputs = transformer_encoder(inputs, head_size=64, num_heads=4, ff_dim=128, dropout=0.1)

    assert outputs.shape == inputs.shape, f"Expected output shape {inputs.shape} but got {outputs.shape}"


def test_transformers_kernel_initialization():
    kernel = TransformersKernel(input_shape=(24, 20), num_classes=2)
    assert isinstance(kernel.model, Model), "TransformersKernel should initialize with a Keras model"


def test_transformers_kernel_fit():
    x_train = np.random.rand(10, 24, 20)  # 10 samples, 24 timesteps, 20 features
    y_train = np.random.randint(0, 2, size=(10,))  # Binary labels for 10 samples
    kernel = TransformersKernel(input_shape=(24, 20), num_classes=2)

    kernel.fit(x_train, y_train, epochs=1, batch_size=2)  # Dummy training
    assert kernel.model is not None, "Model should be fitted without errors"


def test_transformers_kernel_predict():
    x_test = np.random.rand(5, 24, 20)  # 5 samples, 24 timesteps, 20 features
    kernel = TransformersKernel(input_shape=(24, 20), num_classes=2)

    preds = kernel.predict(x_test)
    assert preds.shape == (5, 1), f"Expected output shape (5, 1) but got {preds.shape}"
