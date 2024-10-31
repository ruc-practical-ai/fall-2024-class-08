from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor


def fit_mlp_regressor_pair(x_train, y_low_train, y_high_train):
    low_model = MLPRegressor(random_state=42, mat_iter=200, tol=0.1)
    high_model = MLPRegressor(random_state=42, mat_iter=200, tol=0.1)
    low_model.fit(x_train, y_low_train)
    high_model.fit(x_train, y_high_train)
    return low_model, high_model


def predict_mlp_regressor_pair(low_model, high_model, x_train):
    y_low_train_hat = low_model.predict(x_train)
    y_high_train_hat = high_model.predict(x_train)
    return y_low_train_hat, y_high_train_hat


def fit_linear_regressor_pair(x_train, y_low_train, y_high_train):
    low_model = LinearRegression()
    high_model = LinearRegression()
    low_model.fit(x_train, y_low_train)
    high_model.fit(x_train, y_high_train)
    return low_model, high_model


def predict_linear_regressor_pair(low_model, high_model, x_train):
    y_low_train_hat = low_model.predict(x_train)
    y_high_train_hat = high_model.predict(x_train)
    return y_low_train_hat, y_high_train_hat
