import os
from pathlib import Path

from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler

from experiments.project_6_first_mipha_paper.ask_29_feature_extractors_influence.constants import ALL_LAB_CATEGORIES, \
    RANDOM_SEED, DATA_ROOT
from experiments.project_6_first_mipha_paper.utils.run_experiment import run_experiment
from models.mipha.components.aggregators.horizontal_stack_aggregator import HorizontalStackAggregator
from models.mipha.components.evaluators.classification_evaluator import ClassificationEvaluator
from models.mipha.components.feature_extractors.pass_through_feature_extractor import PassThroughFeatureExtractor
from models.mipha.components.kernels.cnn_inn_kernel import CnnInnKernel

"""
Experiment summary:
    - similar to 58_002
    - using CNN model
    - oversampling down to 0.5

Data: ckd_B24m_G3m_P1y

Model summary: 
    - 3D data, Transformers, RandomOverSampler
    - somewhat small model to avoid overfitting

Observations:
    - Lowering the oversampling seems to have made things even worse
"""

processed_data_path = "../out/data_ckd_B24m_G3m_P1y.pkl"
experiment_name = Path(os.path.basename(__file__)).stem
results_file = os.path.join("results", experiment_name + ".json")

def main():
    run_experiment(
        data_root=DATA_ROOT,
        bgp_string="B24m_G3m_P1y",
        lab_categories=ALL_LAB_CATEGORIES,
        disease_identifier="t2d",
        setup_components_func=setup_components,
        fit_parameters={"epochs": 10, "batch_size": 64, "validation_split": 0.1},
        kept_data_sources=[
            "renal",
            "hepatic_renal",
            "metabolic",
            "cardiology",
            "nutrition",
            "infectiology",
            "immunology_inflammation"
        ],
        save_data_to=processed_data_path,
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
    main()
