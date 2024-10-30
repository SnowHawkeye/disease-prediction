from imblearn.under_sampling import RandomUnderSampler
from keras.src.losses import BinaryFocalCrossentropy
from keras.src.optimizers import Adam
from sklearn.preprocessing import StandardScaler

from experiments.project_6_first_mipha_paper.utils.run_experiment import run_experiment, parse_arguments
from models.mipha.components.aggregators.horizontal_stack_aggregator import HorizontalStackAggregator
from models.mipha.components.evaluators.classification_evaluator import ClassificationEvaluator
from models.mipha.components.feature_extractors.pass_through_feature_extractor import PassThroughFeatureExtractor
from models.mipha.components.kernels.transformers_kernel import TransformersKernel

"""
Experiment with autoencoder feature extraction.

Data: t2d_B24m_G3m_P1y
Model summary: 3D data, Transformers, RandomUnderSampler (0.25 ratio)
Observations: 
"""

RANDOM_SEED = 39
data_path = "../out/data.pickle"


def main(arguments):
    run_experiment(
        arguments=arguments,
        setup_components_func=setup_components,
        fit_parameters={"epochs": 10, "batch_size": 32, "validation_split":0.1},
        kept_data_sources=["lab_data_sources", "ecg_data_sources","demographics_data_sources_3d"],
        save_data_to=data_path,
        random_seed=RANDOM_SEED,
        imputer="auto",
    )


def setup_components(data_sources_train, data_sources_test, labels_train, labels_test):
    print("Setting up MIPHA components...")

    n_timesteps = data_sources_train[0].data.shape[1]
    n_features = sum(data_source.data[0].shape[-1] for data_source in data_sources_train)
    model_input_shape = (n_timesteps, n_features)

    print(f"Computed model input shape: {model_input_shape}")

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

    model = TransformersKernel(
        input_shape=model_input_shape,
        num_classes=2,
        head_size=32,
        num_heads=2,
        ff_dim=64,
        num_transformer_blocks=2,
        mlp_units=[64],
        dropout=0.2,
        mlp_dropout=0.2,
        loss=BinaryFocalCrossentropy(gamma=2., alpha=0.25),
        optimizer=Adam(learning_rate=1e-3),
        metrics=["AUC", "Recall", "Precision"],
        component_name=None,  # default params
        imputer=None,  # imputation is already done when data sources are loaded
        resampler=RandomUnderSampler(random_state=RANDOM_SEED, sampling_strategy=0.25),
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
