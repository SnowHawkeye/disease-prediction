import argparse

from imblearn.over_sampling import RandomOverSampler
from mipha.framework import MiphaPredictor
from sklearn.preprocessing import StandardScaler

from experiments.project_6_first_mipha_paper.utils.load_data_utils import make_data_sources
from models.mipha.components.aggregators.horizontal_stack_aggregator import HorizontalStackAggregator
from models.mipha.components.evaluators.classification_evaluator import ClassificationEvaluator
from models.mipha.components.feature_extractors.cnn_autoencoder_feature_extractor import CnnAutoencoderFeatureExtractor
from models.mipha.components.feature_extractors.pass_through_feature_extractor import PassThroughFeatureExtractor
from models.mipha.components.kernels.random_forest_kernel import RandomForestKernel

RANDOM_SEED = 39

"""
Example experiment with sample data.

Data: sample_data_t2d_B2y_G3m_P1y
"""

def main(arguments):
    data_sources_train, data_sources_test, labels_train, labels_test = setup_datasources(arguments.data_root)
    components = setup_components(
        lab_input_shape=data_sources_train[0].data[0].shape,
        ecg_input_shape=data_sources_train[1].data[0].shape,
    )

    mipha = MiphaPredictor(
        feature_extractors=components['feature_extractors'],
        aggregator=components['aggregator'],
        model=components['model'],
        evaluator=components['evaluator'],
    )

    mipha.fit(
        data_sources=data_sources_train,
        train_labels=labels_train,

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
    data_sources = make_data_sources(arguments=data_root, test_size=0.2, random_seed=RANDOM_SEED)

    lab_datasource_train, lab_datasource_test = data_sources["lab_data_sources"]
    ecg_datasource_train, ecg_datasource_test = data_sources["ecg_data_sources"]
    demographics_datasource_train, demographics_datasource_test = data_sources["demographics_data_sources_2d"]

    data_sources_train = [lab_datasource_train, ecg_datasource_train, demographics_datasource_train]
    data_sources_test = [lab_datasource_test, ecg_datasource_test, demographics_datasource_test]
    labels_train, labels_test = data_sources["labels"]

    return data_sources_train, data_sources_test, labels_train, labels_test


def setup_components(lab_input_shape, ecg_input_shape):
    feature_extractors = [
        CnnAutoencoderFeatureExtractor(
            component_name="lab_extractor",
            managed_data_types=["laboratory"],
            input_shape=lab_input_shape,
            latent_dim=10,
        ),
        CnnAutoencoderFeatureExtractor(
            component_name="ecg_extractor",
            managed_data_types=["ecg"],
            input_shape=ecg_input_shape,
            latent_dim=10,
        ),
        PassThroughFeatureExtractor(
            component_name="demographics_extractor",
            managed_data_types=["demographics"],
        )
    ]

    aggregator = HorizontalStackAggregator()

    model = RandomForestKernel(
        resampler=RandomOverSampler(random_state=RANDOM_SEED),
        scaler=StandardScaler(),
        random_state=RANDOM_SEED,
    )

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
    parser.add_argument('--save_model', type=str, required=False, help='Path to save the trained model')
    parser.add_argument('--save_results', type=str, required=False, help='Path to save the classification results')

    # Parse the arguments
    args = parser.parse_args()
    main(args)
