import numpy as np
import pandas as pd
import seaborn as sns
import tsfel
from imblearn.over_sampling import RandomOverSampler
from keras import Sequential
from keras.src.layers import Conv1D, BatchNormalization, Activation, Flatten, Dense
from matplotlib import pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tqdm.notebook import tqdm

from mipha.framework import MachineLearningModel, FeatureExtractor, Aggregator, Evaluator


# UTILITY FUNCTIONS
def make_simple_imputer(strategy="mean"):
    return SimpleImputer(keep_empty_features=True, strategy=strategy)


def impute_data(x, data_imputer):
    """
    Performs data imputation on a per-matrix basis.
    :param x: list of history matrices
    :param data_imputer: the imputer to use to replace missing data
    :return: a list of history matrices with imputed data
    """
    print("Imputing data...")
    data_x_imputed = []
    for matrix in tqdm(x):
        matrix_imputed = data_imputer.fit_transform(matrix)
        data_x_imputed.append(matrix_imputed)
    print("Data successfully imputed!")
    return data_x_imputed


def scale_data_train(train_data, train_scaler):
    x_train_shape = np.array(train_data).shape
    num_analyses = x_train_shape[-1]

    print("Scaling x_train...")

    x_train_reshape = np.stack(train_data).reshape(-1, num_analyses)
    train_scaler.fit(x_train_reshape)

    x_train_reshape = train_scaler.transform(x_train_reshape)
    x_train_final = x_train_reshape.reshape(x_train_shape)

    print("x_train scaled successfully!")

    return x_train_final


def scale_data_test(test_data, trained_scaler):
    x_test_shape = np.array(test_data).shape
    num_analyses = x_test_shape[-1]

    print("Scaling x_test...")

    x_test_reshape = np.stack(test_data).reshape(-1, num_analyses)

    x_test_reshape = trained_scaler.transform(x_test_reshape)
    x_test_final = x_test_reshape.reshape(x_test_shape)

    print("x_test scaled successfully!")

    return x_test_final


def cnn_model(rows, columns, output_dim, n_filters):
    cnn = Sequential()
    cnn.add(Conv1D(filters=n_filters, kernel_size=3, input_shape=(columns, rows)))
    cnn.add(BatchNormalization())
    cnn.add(Activation("relu"))
    cnn.add(Flatten())
    cnn.add(Dense(32, activation='relu'))
    cnn.add(Dense(output_dim, activation='sigmoid'))

    print("Compiling the model...")
    cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print("Model compiled!")
    return cnn


def evaluate_model(model: MachineLearningModel, x_test, y_test, threshold):
    y_pred = model.predict(x_test)
    y_pred_binary = pd.DataFrame(y_pred).apply(lambda val: (val > threshold).astype(int))

    # Metrics
    print("Test Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred_binary) * 100))
    print("ROC-AUC: {:.2f}%".format(roc_auc_score(y_test, y_pred_binary) * 100))
    print(classification_report(y_test, y_pred_binary))

    # Confusion matrix
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    sns.heatmap(confusion_matrix(y_test, y_pred_binary, normalize="true"), annot=True)
    plt.subplot(122)
    sns.heatmap(confusion_matrix(y_test, y_pred_binary, normalize=None), annot=True)
    plt.show()
    print(confusion_matrix(y_test, y_pred_binary, normalize="true"))


# FRAMEWORK IMPLEMENTATION

class BiologyFeatureExtractor(FeatureExtractor):
    def __init__(self, component_name, managed_data_types: list[str] = None):
        super().__init__(component_name=component_name, managed_data_types=managed_data_types)

    def extract_features(self, x):
        config = tsfel.get_features_by_domain()
        simple_imputer = make_simple_imputer()
        data_imputed = impute_data(x, simple_imputer)
        extracted_features = tsfel.time_series_features_extractor(config, data_imputed).astype(float)
        extracted_features.reset_index(drop=True, inplace=True)
        return extracted_features


class DemographicsFeatureExtractor(FeatureExtractor):
    def __init__(self, component_name, managed_data_types: list[str] = None):
        super().__init__(component_name=component_name, managed_data_types=managed_data_types)

    def extract_features(self, x):
        one_hot_encoder = OneHotEncoder(drop="if_binary")
        one_hot_encoder.fit(x.loc[:, ["gender"]])  # keep the column as a DataFrame (necessary for one-hot encoding)
        data_gender_onehot = one_hot_encoder.transform(x.loc[:, ["gender"]]).toarray()
        extracted_features = pd.DataFrame(data_gender_onehot, columns=one_hot_encoder.get_feature_names_out())
        extracted_features["age"] = x["age"].astype("Int64")
        extracted_features.reset_index(drop=True, inplace=True)
        return extracted_features


class SimpleAggregator(Aggregator):
    def aggregate_features(self, features):
        aggregated_features = pd.concat([t for t in features], axis=1)
        return aggregated_features


class SimpleCnnModel(MachineLearningModel):
    def __init__(self, rows, columns, output_dim, n_filters):
        super().__init__()
        self.imputer = make_simple_imputer()
        self.scaler = StandardScaler()
        self.resampler = RandomOverSampler(random_state=25, sampling_strategy=0.5)
        self.model = cnn_model(rows, columns, output_dim, n_filters)
        self.model.summary()

    def fit(self, x, y, *args, **kwargs):
        _x_imputed = self.imputer.fit_transform(x)  # impute missing data
        _x_scaled = scale_data_train(_x_imputed, self.scaler)  # scale data
        _x_train, _y_train = self.resampler.fit_resample(_x_scaled, y)  # resample data
        _y_train = np.array(_y_train)
        self.model.fit(_x_train, _y_train, *args, **kwargs)
        print("Model fit successfully!")
        return self.model

    def predict(self, x, *args, **kwargs):
        _x_processed = self.process_data(x)
        return self.model.predict(_x_processed, *args, **kwargs)

    def process_data(self, x):
        _x_imputed = self.imputer.transform(x)  # impute missing data
        _x_scaled = scale_data_test(_x_imputed, self.scaler)  # scale data
        return _x_scaled


class SimpleEvaluator(Evaluator):
    def evaluate_model(self, model: MachineLearningModel, x_test, y_test, threshold=0.5, *args, **kwargs):
        return evaluate_model(model, x_test, y_test, threshold)
