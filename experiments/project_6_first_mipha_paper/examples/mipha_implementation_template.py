import argparse

from mipha.framework import MiphaPredictor

from experiments.project_6_first_mipha_paper.utils.load_data_utils import make_data_sources
from models.mipha.components.aggregators.horizontal_stack_aggregator import HorizontalStackAggregator
from models.mipha.components.evaluators.classification_evaluator import ClassificationEvaluator


def main(arguments):
    components = setup_components()

    mipha = MiphaPredictor(
        feature_extractors=components['feature_extractors'],
        aggregator=components['aggregator'],
        model=components['model'],
        evaluator=components['evaluator'],
    )

    data_sources_train, data_sources_test, labels_train, labels_test = setup_datasources(arguments.data_root)

    mipha.fit(
        data_sources=data_sources_train,
        train_labels=labels_train,
        epochs=...
    )

    if arguments.save_model:
        mipha.save(arguments.save_model)

    mipha.evaluate(
        data_sources=data_sources_test,
        test_labels=labels_test,
    )

    if arguments.save_results:
        mipha.evaluator.save_metrics_to_json(arguments.save_results)

    mipha.evaluator.display_results()


def setup_datasources(data_root):
    data_sources = make_data_sources(data_root=data_root, test_size=0.2, random_seed=39)

    data_sources_train = [...]
    data_sources_test = [...]
    labels_train, labels_test = data_sources["labels"]

    return data_sources_train, data_sources_test, labels_train, labels_test


def setup_components():
    feature_extractors = [...]
    aggregator = HorizontalStackAggregator()
    model = ...
    evaluator = ClassificationEvaluator()

    return {
        "feature_extractors": feature_extractors,
        "aggregator": aggregator,
        "model": model,
        "evaluator": evaluator,
    }


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Load data sources for model training and testing.")

    # Add argument for data_root
    parser.add_argument('--data_root', type=str, required=True, help='Path to the root directory of the data')
    parser.add_argument('--save_model', type=str, required=True, help='Path to save the trained model')
    parser.add_argument('--save_results', type=str, required=True, help='Path to save the classification results')

    # Parse the arguments
    args = parser.parse_args()
    main(args)
