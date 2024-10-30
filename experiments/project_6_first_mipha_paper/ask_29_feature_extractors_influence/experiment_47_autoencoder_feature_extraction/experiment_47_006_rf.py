from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler

from experiments.project_6_first_mipha_paper.utils.run_experiment import run_experiment, parse_arguments
from models.mipha.components.aggregators.horizontal_stack_aggregator import HorizontalStackAggregator
from models.mipha.components.evaluators.classification_evaluator import ClassificationEvaluator
from models.mipha.components.feature_extractors.cnn_autoencoder_feature_extractor import CnnAutoencoderFeatureExtractor
from models.mipha.components.feature_extractors.pass_through_feature_extractor import PassThroughFeatureExtractor
from models.mipha.components.kernels.random_forest_kernel import RandomForestKernel

"""
Experiment with autoencoder feature extraction.

Data: t2d_B24m_G3m_P1y
Model summary: Random forest, RandomOverSampler, attempt to reduce overfitting
Observations: 
"""

RANDOM_SEED = 39
data_path = "../out/data.pickle"


def main(arguments):
    run_experiment(
        arguments=arguments, setup_components_func=setup_components,
        fit_parameters=None,
        kept_data_sources=["lab_data_sources", "ecg_data_sources", "demographics_data_sources_2d"],
        save_data_to=data_path,
        random_seed=RANDOM_SEED,
    )


def setup_components(data_sources_train, data_sources_test, labels_train, labels_test):
    n_timesteps = data_sources_train[0].data.shape[1]
    n_features_lab = data_sources_train[0].data.shape[-1]
    n_features_ecg = data_sources_train[1].data.shape[-1]

    print("Setting up MIPHA components...")

    latent_dim_lab = 10
    latent_dim_ecg = 5

    lab_feature_extractor = CnnAutoencoderFeatureExtractor(
        input_shape=(n_timesteps, n_features_lab),
        latent_dim=latent_dim_lab,
        n_layers=2,
        n_filters=64,
        kernel_size=3,
        strides=1,
        activation='relu',
        loss='mse',
        optimizer='adam',
        component_name="lab_extractor",
        managed_data_types=["laboratory"],
    )

    ecg_feature_extractor = CnnAutoencoderFeatureExtractor(
        input_shape=(n_timesteps, n_features_ecg),
        latent_dim=latent_dim_ecg,
        n_layers=2,
        n_filters=64,
        kernel_size=3,
        strides=1,
        activation='relu',
        loss='mse',
        optimizer='adam',
        component_name="ecg_extractor",
        managed_data_types=["ecg"],
    )

    demographics_feature_extractor = PassThroughFeatureExtractor(
        component_name="demographics_extractor",
        managed_data_types=["demographics"],
    )

    print("Fitting feature extractors...")
    lab_feature_extractor.fit(data_sources_train[0].data, epochs=10, batch_size=124)
    ecg_feature_extractor.fit(data_sources_train[1].data, epochs=10, batch_size=124)

    feature_extractors = [lab_feature_extractor, ecg_feature_extractor, demographics_feature_extractor]

    aggregator = HorizontalStackAggregator()

    model = RandomForestKernel(
        n_estimators=50,  # Reduced number of trees
        max_depth=10,  # Limit the depth of trees
        min_samples_split=5,  # Require more samples to split
        min_samples_leaf=2,  # Require more samples at leaf nodes
        max_features='sqrt',  # Use sqrt of the number of features
        random_state=RANDOM_SEED,  # For reproducibility
        verbose=1,
        component_name=None,  # default params
        imputer=None,  # imputation is already done when data sources are loaded
        resampler=RandomOverSampler(random_state=RANDOM_SEED, sampling_strategy=1),
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
