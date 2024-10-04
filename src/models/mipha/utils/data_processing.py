from sklearn.impute import SimpleImputer
from tqdm import tqdm
import numpy as np


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
