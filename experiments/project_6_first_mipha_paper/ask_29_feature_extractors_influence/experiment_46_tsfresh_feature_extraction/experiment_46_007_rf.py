import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from tsfresh.feature_extraction import MinimalFCParameters

from experiments.project_6_first_mipha_paper.utils.run_experiment import run_experiment, parse_arguments
from models.mipha.components.aggregators.horizontal_stack_aggregator import HorizontalStackAggregator
from models.mipha.components.evaluators.classification_evaluator import ClassificationEvaluator
from models.mipha.components.feature_extractors.pass_through_feature_extractor import PassThroughFeatureExtractor
from models.mipha.components.feature_extractors.tsfresh_feature_extractor import TsfreshFeatureExtractor
from models.mipha.components.kernels.random_forest_kernel import RandomForestKernel

"""
Experiment with tsfresh feature extraction.

Data: t2d_B24m_G3m_P1y
Model summary: Random forest, SMOTE
Observations: 
"""

RANDOM_SEED = 39
data_path = "../out/data.pickle"


def compute_class_weight(n_samples, n_classes, n_samples_in_class):
    return n_samples / (n_samples_in_class * n_classes)


def main(arguments):
    run_experiment(
        arguments=arguments, setup_components_func=setup_components,
        fit_parameters=None,
        kept_data_sources=["lab_data_sources", "ecg_data_sources", "demographics_data_sources_2d"],
        save_data_to=data_path,
        random_seed=RANDOM_SEED,
    )


def setup_components(data_sources_train, data_sources_test, labels_train, labels_test):
    print("Setting up MIPHA components...")

    lab_feature_extractor = TsfreshFeatureExtractor(
        component_name="lab_extractor",
        managed_data_types=["laboratory"],
        default_fc_parameters=MinimalFCParameters(),
    )

    ecg_feature_extractor = TsfreshFeatureExtractor(
        component_name="ecg_extractor",
        managed_data_types=["ecg"],
        default_fc_parameters=MinimalFCParameters(),
    )

    demographics_feature_extractor = PassThroughFeatureExtractor(
        component_name="demographics_extractor",
        managed_data_types=["demographics"],
    )

    feature_extractors = [lab_feature_extractor, ecg_feature_extractor, demographics_feature_extractor]

    aggregator = HorizontalStackAggregator()

    n_samples = len(data_sources_train)
    counts = np.bincount(labels_train)
    count_0 = counts[0]  # Count of 0
    count_1 = counts[1]  # Count of 1

    class_weight = {
        0: compute_class_weight(n_samples=n_samples, n_classes=2, n_samples_in_class=count_0),
        1: compute_class_weight(n_samples=n_samples, n_classes=2, n_samples_in_class=count_1)
    }
    model = RandomForestKernel(
        n_estimators=100,  # Number of trees in the forest
        max_depth=None,  # No limit on tree depth (allow full growth)
        min_samples_split=2,  # Minimum number of samples required to split a node
        min_samples_leaf=1,  # Minimum number of samples required to be at a leaf node
        max_features='sqrt',  # Use the square root of the number of features
        bootstrap=True,  # Use bootstrap sampling
        class_weight=class_weight,
        verbose=1,
        random_state=RANDOM_SEED,
        component_name=None,  # default params
        imputer=None,  # imputation is already done when data sources are loaded
        resampler=SMOTE(random_state=RANDOM_SEED, sampling_strategy=0.8),
        scaler=StandardScaler(),
    )

    evaluator = ClassificationEvaluator()

    return {
        "feature_extractors": feature_extractors,
        "aggregator": aggregator,
        "model": model,
        "evaluator": evaluator,
    }


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
