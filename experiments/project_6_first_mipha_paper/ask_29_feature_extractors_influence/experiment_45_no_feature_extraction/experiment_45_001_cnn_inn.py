import argparse
import os

from imblearn.over_sampling import RandomOverSampler
from mipha.framework import MiphaPredictor
from sklearn.preprocessing import StandardScaler

from experiments.project_6_first_mipha_paper.utils.load_data_utils import make_data_sources
from features.mimic.extract_lab_records import save_pickle, load_pickle
from models.mipha.components.aggregators.horizontal_stack_aggregator import HorizontalStackAggregator
from models.mipha.components.evaluators.classification_evaluator import ClassificationEvaluator
from models.mipha.components.feature_extractors.pass_through_feature_extractor import PassThroughFeatureExtractor
from models.mipha.components.kernels.cnn_inn_kernel import CnnInnKernel

"""
Experiment without feature extraction.

Data: t2d_B24m_G3m_P1y
Model summary: CNN-INN kernel (3 layers of each)
"""

RANDOM_SEED = 39
data_path = "../out/test_data.pickle"


def main(arguments):
    if not os.path.exists(data_path):
        data_sources_train, data_sources_test, labels_train, labels_test = setup_datasources(arguments.data_root)

        print("Saving data sources...")
        save_pickle((data_sources_train, data_sources_test, labels_train, labels_test), data_path)

    else:  # if the datasource setup has already been saved
        print("Pre-processed data sources have been found. Loading...")
        data_sources_train, data_sources_test, labels_train, labels_test = load_pickle(data_path)

    # no feature extraction, input shape is determined by the sum of the number of features
    n_timesteps = data_sources_train[0].data.shape[1]
    n_features = sum(data_source.data[0].shape[-1] for data_source in data_sources_train)
    components = setup_components(model_input_shape=(n_timesteps, n_features))

    mipha = MiphaPredictor(
        feature_extractors=components['feature_extractors'],
        aggregator=components['aggregator'],
        model=components['model'],
        evaluator=components['evaluator'],
    )

    mipha.fit(
        data_sources=data_sources_train,
        train_labels=labels_train,
        epochs=20,
        batch_size=128
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
    print("Loading data sources...")
    data_sources = make_data_sources(arguments=data_root, test_size=0.2, random_seed=RANDOM_SEED)

    lab_datasource_train, lab_datasource_test = data_sources["lab_data_sources"]
    ecg_datasource_train, ecg_datasource_test = data_sources["ecg_data_sources"]
    demographics_datasource_train, demographics_datasource_test = data_sources["demographics_data_sources_3d"]

    data_sources_train = [lab_datasource_train, ecg_datasource_train, demographics_datasource_train]
    data_sources_test = [lab_datasource_test, ecg_datasource_test, demographics_datasource_test]
    labels_train, labels_test = data_sources["labels"]

    return data_sources_train, data_sources_test, labels_train, labels_test


def setup_components(model_input_shape):
    print("Setting up MIPHA components...")

    feature_extractors = [
        PassThroughFeatureExtractor(
            component_name="lab_extractor",
            managed_data_types=["laboratory"],
        ),
        PassThroughFeatureExtractor(
            component_name="ecg_extractor",
            managed_data_types=["ecg"],
        ),
        PassThroughFeatureExtractor(
            component_name="demographics_extractor",
            managed_data_types=["demographics"],
        )
    ]

    aggregator = HorizontalStackAggregator()

    model = CnnInnKernel(
        input_shape=model_input_shape,
        num_classes=2,
        num_convolution_layers=3,
        num_involution_layers=3,
        convolution_params=None,  # default params
        involution_params=None,  # default params
        loss=None,  # default params
        optimizer='adam',
        metrics=None,  # default params
        component_name=None,  # default params
        imputer=None,  # imputation is already done when data sources are loaded
        resampler=RandomOverSampler(random_state=RANDOM_SEED, sampling_strategy=0.5),
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
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Load data sources for model training and testing.")

    # Add argument for data_root
    parser.add_argument('--data_root', type=str, required=True, help='Path to the root directory of the data')
    parser.add_argument('--save_model', type=str, required=False, help='Path to save the trained model')
    parser.add_argument('--save_results', type=str, required=False, help='Path to save the classification results')

    # Parse the arguments
    args = parser.parse_args()
    main(args)
