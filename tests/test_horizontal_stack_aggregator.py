import numpy as np
import pytest

from src.models.mipha.components.aggregators.horizontal_stack_aggregator import HorizontalStackAggregator


@pytest.fixture
def aggregator():
    return HorizontalStackAggregator()


def test_aggregate_features_all_3d_inputs(aggregator):
    arr1 = np.random.rand(100, 5, 8)
    arr2 = np.random.rand(100, 5, 6)

    result = aggregator.aggregate_features([arr1, arr2])
    assert result.shape == (100, 5, 14)


def test_aggregate_features_all_2d_inputs(aggregator):
    arr1 = np.random.rand(100, 10)
    arr2 = np.random.rand(100, 15)

    result = aggregator.aggregate_features([arr1, arr2])
    assert result.shape == (100, 25)


def test_aggregate_features_mixed_inputs(aggregator):
    arr1 = np.random.rand(100, 5, 8)
    arr2 = np.random.rand(100, 10)

    result = aggregator.aggregate_features([arr1, arr2])
    assert result.shape == (100, 5 * 8 + 10)


def test_aggregate_features_different_sample_sizes(aggregator):
    arr1 = np.random.rand(100, 5, 8)
    arr2 = np.random.rand(200, 5, 6)

    with pytest.raises(ValueError, match="All feature arrays must have the same number of samples"):
        aggregator.aggregate_features([arr1, arr2])


def test_aggregate_features_single_input(aggregator):
    arr1 = np.random.rand(100, 5, 8)

    result = aggregator.aggregate_features([arr1])
    assert np.array_equal(result, arr1)
