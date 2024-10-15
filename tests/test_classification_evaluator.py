import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from mipha.framework import MachineLearningModel

from models.mipha.components.evaluators.classification_evaluator import ClassificationEvaluator


# Sample data for testing
@pytest.fixture
def sample_data():
    x_test = np.array([[0.1, 0.2], [0.2, 0.1], [0.1, 0.4], [0.5, 0.6]])
    y_test_binary = np.array([0, 1, 0, 1])
    y_test_multiclass = np.array([0, 1, 2, 1])
    return x_test, y_test_binary, y_test_multiclass


@pytest.fixture
def mock_model():
    model = MagicMock(spec=MachineLearningModel)
    return model


def test_evaluate_model_binary(mock_model, sample_data):
    x_test, y_test, _ = sample_data
    # Set predictions for the mock model
    mock_model.predict.return_value = np.array([[0.4], [0.6], [0.3], [0.8]])

    evaluator = ClassificationEvaluator()
    metrics = evaluator.evaluate_model(mock_model, x_test, y_test)

    # Assertions
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1_score"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert "classification_report" in metrics
    assert "raw_confusion_matrix" in metrics
    assert "normalized_confusion_matrix" in metrics


def test_evaluate_model_multiclass(mock_model, sample_data):
    x_test, _, y_test = sample_data
    # Set predictions for the mock model
    mock_model.predict.return_value = np.array([[0.1, 0.5, 0.4], [0.3, 0.4, 0.3], [0.2, 0.1, 0.7], [0.5, 0.4, 0.1]])

    evaluator = ClassificationEvaluator()
    metrics = evaluator.evaluate_model(mock_model, x_test, y_test)

    # Assertions
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1_score"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert "classification_report" in metrics
    assert "raw_confusion_matrix" in metrics
    assert "normalized_confusion_matrix" in metrics


def test_save_metrics_to_json(sample_data):
    evaluator = ClassificationEvaluator()
    metrics = {
        "accuracy": 0.75,
        "f1_score": 0.7,
        "precision": 0.8,
        "recall": 0.6,
        "classification_report": {}
    }
    evaluator.metrics = metrics

    with patch("builtins.open", new_callable=MagicMock) as mock_open:
        evaluator.save_metrics_to_json("metrics.json")
        mock_open.assert_called_once_with("metrics.json", 'w')


def test_display_results(sample_data):
    evaluator = ClassificationEvaluator()
    metrics = {
        "accuracy": 0.75,
        "f1_score": 0.7,
        "precision": 0.8,
        "recall": 0.6,
        "classification_report": {},
        "raw_confusion_matrix": {},
        "normalized_confusion_matrix": {},
    }
    evaluator.metrics = metrics

    x_test, y_test, _ = sample_data
    y_pred_classes = np.array([0, 1, 0, 1])

    with patch("matplotlib.pyplot.show") as mock_show:
        with patch("seaborn.heatmap") as mock_heatmap:
            evaluator.display_results(y_test, y_pred_classes)
            mock_show.assert_called_once()
            mock_heatmap.assert_called()
