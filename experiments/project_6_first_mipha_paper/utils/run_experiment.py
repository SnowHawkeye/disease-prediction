import argparse
from datetime import datetime
from typing import Callable

from mipha.framework import MiphaPredictor

from experiments.project_6_first_mipha_paper.utils.load_data_utils import load_data_sources, setup_data_sources


def run_experiment(
        arguments,
        setup_components_func: Callable,
        kept_data_sources: list,
        save_data_to,
        random_seed: int,
        fit_parameters: dict = None,
        imputer="auto",
):
    print(f"Computation started at {datetime.now()}")
    if fit_parameters is None:
        fit_parameters = {}

    data_sources = load_data_sources(arguments.data_root, save_data_to, random_seed=random_seed, imputer=imputer)
    data_sources_train, data_sources_test, labels_train, labels_test = setup_data_sources(
        data_sources=data_sources,
        kept_data_sources=kept_data_sources
    )

    print(f"Shapes for train datasets: {[t.data.shape for t in data_sources_train]}")
    print(f"Shapes for test datasets: {[t.data.shape for t in data_sources_test]}")

    components = setup_components_func(data_sources_train, data_sources_test, labels_train, labels_test)

    mipha = MiphaPredictor(
        feature_extractors=components['feature_extractors'],
        aggregator=components['aggregator'],
        model=components['model'],
        evaluator=components['evaluator'],
    )

    mipha.fit(
        data_sources=data_sources_train,
        train_labels=labels_train,
        **fit_parameters,
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
    print(f"Computation finished at {datetime.now()}")


def parse_arguments():
    global args
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Load data sources for model training and testing.")
    # Add argument for data_root
    parser.add_argument('--data_root', type=str, required=True, help='Path to the root directory of the data')
    parser.add_argument('--save_model', type=str, required=False, help='Path to save the trained model')
    parser.add_argument('--save_results', type=str, required=False, help='Path to save the classification results')
    # Parse the arguments
    args = parser.parse_args()
    return args
