import pytest
from keras import backend as K
from keras import Model

from models.mipha.components.kernels.cnn_inn_kernel import build_model


@pytest.fixture
def cleanup():
    """Fixture to clear the Keras session after tests."""
    yield
    K.clear_session()


def test_build_model_with_default_params(cleanup):
    model = build_model((24, 20))
    assert isinstance(model, Model)
    assert model.input_shape == (None, 24, 20)
    assert model.output_shape == (None, 1)
    assert len(model.layers) == 9  # Input, Conv1D, Dropout, Reshape, Involution, Dropout, Reshape, GlobalAveragePooling, Output


def test_build_model_with_convolution_layers(cleanup):
    model = build_model((24, 20), num_convolution_layers=2)
    assert isinstance(model, Model)
    assert model.input_shape == (None, 24, 20)
    assert len(model.layers) == 11  # 5 + 2 * Conv1D/Dropout + 1 * Involution/Dropout


def test_build_model_with_involution_layers(cleanup):
    model = build_model((24, 20), num_involution_layers=2)
    assert isinstance(model, Model)
    assert model.input_shape == (None, 24, 20)
    assert len(model.layers) == 11  # 5 + 1 * Conv1D/Dropout + 2 * Involution/Dropout


def test_build_model_with_no_convolution_layers(cleanup):
    model = build_model((24, 20), num_convolution_layers=0, num_involution_layers=2)
    assert isinstance(model, Model)
    assert model.input_shape == (None, 24, 20)
    assert len(model.layers) == 9  # 5 + 0 * Conv1D/Dropout + 2 * Involution/Dropout


def test_build_model_with_no_involution_layers(cleanup):
    model = build_model((24, 20), num_convolution_layers=2, num_involution_layers=0)
    assert isinstance(model, Model)
    assert model.input_shape == (None, 24, 20)
    assert len(model.layers) == 9  # 5 + 2 * Conv1D/Dropout + 0 * Involution/Dropout


def test_build_model_with_no_layers(cleanup):
    model = build_model((24, 20), num_convolution_layers=0, num_involution_layers=0)
    assert isinstance(model, Model)
    assert model.input_shape == (None, 24, 20)
    assert model.output_shape == (None, 1)
    assert len(model.layers) == 5  # 5 + 0 * Conv1D/Dropout + 0 * Involution/Dropout


def test_build_model_with_multi_class(cleanup):
    model = build_model((24, 20), num_classes=5)
    assert isinstance(model, Model)
    assert model.output_shape == (None, 5)
