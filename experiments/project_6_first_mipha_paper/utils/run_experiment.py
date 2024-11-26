from datetime import datetime
from typing import Callable

from mipha.framework import MiphaPredictor

from experiments.project_6_first_mipha_paper.utils.load_data_utils import load_data_sources, setup_data_sources


def run_experiment(
        data_root,
        bgp_string,
        lab_categories,
        disease_identifier,
        setup_components_func: Callable,
        kept_data_sources: list,
        save_data_to,
        random_seed: int,
        save_results=None,
        save_model=None,
        fit_parameters: dict = None,
        imputer="auto",
):
    """
    :param data_root: root directory of the data
    :param bgp_string: backward window, gap and prediction window as a string (formatted like in the paths)
    :param lab_categories: laboratory data categories to be fetched
    :param disease_identifier: string representing the disease as used in the paths
    :param setup_components_func: function called to set up MIPHA components
    :param kept_data_sources: name of the data sources to use in the experiment
    :param save_data_to: directory to save the matrices to (as pickle, parent directory created if it doesn't exist)
    :param save_results: file to save the results to
    :param save_model: file to save the trained MIPHA model to
    :param random_seed: seed for the random number generator
    :param fit_parameters: parameters to pass to the fit function
    :param imputer: imputer used to impute each individual matrix of the time series (its state is reset for each matrix)
    :return:
    """
    print(f"Computation started at {datetime.now()}")
    if fit_parameters is None:
        fit_parameters = {}

    data_sources = load_data_sources(
        root_dir=data_root,
        bgp_string=bgp_string,
        lab_categories=lab_categories,
        disease_identifier=disease_identifier,
        save_to=save_data_to,
        random_seed=random_seed,
        imputer=imputer
    )
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

    if save_model:
        mipha.save(save_model)

    mipha.evaluate(
        data_sources=data_sources_test,
        test_labels=labels_test,
    )

    if save_results:
        mipha.evaluator.save_metrics_to_json(save_results)

    mipha.evaluator.display_results()
    print(f"Computation finished at {datetime.now()}")
