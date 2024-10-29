import numpy as np


def compute_rmse(y_true, y_predicted):
    return np.sqrt(np.average((y_true - y_predicted) ** 2))


def compute_rmse_pair(
    y_low_train, y_high_train, y_low_train_hat, y_high_train_hat
):
    low_rmse_train = compute_rmse(y_low_train, y_low_train_hat)
    high_rmse_train = compute_rmse(y_high_train, y_high_train_hat)
    return low_rmse_train, high_rmse_train
