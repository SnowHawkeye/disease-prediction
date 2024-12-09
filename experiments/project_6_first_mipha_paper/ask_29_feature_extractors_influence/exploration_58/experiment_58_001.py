import os
from pathlib import Path

from imblearn.over_sampling import RandomOverSampler
from keras.src.losses import BinaryFocalCrossentropy
from keras.src.optimizers import Adam
from sklearn.preprocessing import StandardScaler

from experiments.project_6_first_mipha_paper.ask_29_feature_extractors_influence.constants import ALL_LAB_CATEGORIES, \
    RANDOM_SEED, DATA_ROOT
from experiments.project_6_first_mipha_paper.utils.run_experiment import run_experiment
from models.mipha.components.aggregators.horizontal_stack_aggregator import HorizontalStackAggregator
from models.mipha.components.evaluators.classification_evaluator import ClassificationEvaluator
from models.mipha.components.feature_extractors.pass_through_feature_extractor import PassThroughFeatureExtractor
from models.mipha.components.kernels.transformers_kernel import TransformersKernel

"""
Experiment summary:
    - first attempt at using several categories for laboratory data

Data: t2d_B24m_G3m_P1y

Model summary: 
    - 3D data, Transformers, RandomOverSampler
    - somewhat small model to avoid overfitting
    
Observations:
    - results are sadly similar to predictions with the most common analyses
    - generalization on the minority class seems difficult
"""

processed_data_path = "../out/data_t2d_B24m_G3m_P1y.pkl"

experiment_name = Path(os.path.basename(__file__)).stem
results_file = os.path.join("results", experiment_name + ".json")


def main():
    run_experiment(
        data_root=DATA_ROOT,
        bgp_string="B24m_G3m_P1y",
        lab_categories=ALL_LAB_CATEGORIES,
        disease_identifier="t2d",
        setup_components_func=setup_components,
        fit_parameters={"epochs": 3, "batch_size": 64, "validation_split": 0.1},
        kept_data_sources=[
            "metabolic",
            "endocrine",
            "renal",
            "nutrition",
            "cardiology",
            "immunology_inflammation"
        ],
        save_data_to=processed_data_path,
        save_results=results_file,
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
        dropout=0.3,
        mlp_dropout=0.3,
        loss=BinaryFocalCrossentropy(gamma=2., alpha=0.25),
        optimizer=Adam(learning_rate=1e-3),
        metrics=["AUC", "Recall", "Precision"],
        component_name=None,  # default params
        imputer=None,  # imputation is already done when data sources are loaded
        resampler=RandomOverSampler(random_state=RANDOM_SEED, sampling_strategy=1.0),
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
    main()