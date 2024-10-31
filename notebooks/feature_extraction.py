import numpy as np


def get_individual_stock_df(df, symbol):
    stock_df = df[df["Symbol"] == symbol].copy()
    return stock_df


def get_individual_stock_numpy(df, symbol):
    stock_df = get_individual_stock_df(df, symbol)
    stock_array = stock_df[["Low", "High"]].to_numpy()
    return stock_array


def extract_list_of_stock_arrays(stocks_df, predictor_symbols):
    predictor_stock_arrays = []
    for symbol in predictor_symbols:
        predictor_stock_numpy = get_individual_stock_numpy(stocks_df, symbol)
        predictor_stock_arrays.append(predictor_stock_numpy)
    return predictor_stock_arrays


def get_autoregression_targets(
    dates, stocks_df, target_symbol, n_days_history, n_days_forward
):
    target_stock_array = get_individual_stock_numpy(stocks_df, target_symbol)
    target_stock_low = target_stock_array[:, 0]
    target_stock_high = target_stock_array[:, 1]
    start_idx = n_days_history + n_days_forward - 1
    y_targets_low = target_stock_low[start_idx:]
    y_targets_high = target_stock_high[start_idx:]
    dates = dates[start_idx:]
    return dates, y_targets_low, y_targets_high


def generate_flat_auto_regression_feature_array(
    dates, n_days_history, n_days_forward, prediction_array
):
    flat_samples = []
    total_days = dates.shape[0]
    for start_idx in np.arange(
        total_days - n_days_history - n_days_forward + 1
    ):
        end_idx = start_idx + n_days_history
        square_sample = prediction_array[start_idx:end_idx, :]
        flat_sample = square_sample.flatten()
        flat_samples.append(flat_sample)
    x_features = np.vstack(flat_samples)
    return x_features


def get_autoregression_features(
    dates, stocks_df, n_days_history, n_days_forward, predictor_symbols
):

    predictor_stock_arrays = extract_list_of_stock_arrays(
        stocks_df, predictor_symbols
    )
    prediction_array = np.concatenate(predictor_stock_arrays, axis=1)
    x_features = generate_flat_auto_regression_feature_array(
        dates, n_days_history, n_days_forward, prediction_array
    )
    return x_features


def make_train_test_split(variable):
    total_days = len(variable)
    train_split = 0.8
    n_train_days = int(total_days * train_split)
    variable_train = variable[:n_train_days]
    variable_test = variable[n_train_days:]
    return variable_train, variable_test
