import numpy as np


def mean_squared_error(y_pred, y_actual):
    """
    Calculate the Mean Squared Error (MSE) for regression.

    Args:
        y_pred (array-like): Predicted target values.
        y_actual (array-like): Actual target values.

    Returns:
        float: Mean Squared Error.

    Formula:
        MSE = (1/n) * sum((y_pred - y_actual)^2)
    """
    if len(y_pred) != len(y_actual):
        raise ValueError("The lengths of y_pred and y_actual do not match.")

    mse = np.mean((y_pred - y_actual) ** 2)
    return mse


def root_mean_squared_error(y_pred, y_actual):
    """
    Calculate the Root Mean Squared Error (RMSE) for regression.

    Args:
        y_pred (array-like): Predicted target values.
        y_actual (array-like): Actual target values.

    Returns:
        float: Root Mean Squared Error.

    Formula:
        RMSE = sqrt(MSE)
    """
    if len(y_pred) != len(y_actual):
        raise ValueError("The lengths of y_pred and y_actual do not match.")

    mse = np.mean((y_pred - y_actual) ** 2)
    rmse = np.sqrt(mse)
    return rmse


def mean_absolute_error(y_pred, y_actual):
    """
    Calculate the Mean Absolute Error (MAE) for regression.

    Args:
        y_pred (array-like): Predicted target values.
        y_actual (array-like): Actual target values.

    Returns:
        float: Mean Absolute Error.

    Formula:
        MAE = (1/n) * sum(abs(y_pred - y_actual))
    """
    if len(y_pred) != len(y_actual):
        raise ValueError("The lengths of y_pred and y_actual do not match.")

    mae = np.mean(np.abs(y_pred - y_actual))
    return mae


def r_squared(y_pred, y_actual):
    """
    Calculate the R-squared (coefficient of determination) for regression.

    Args:
        y_pred (array-like): Predicted target values.
        y_actual (array-like): Actual target values.

    Returns:
        float: R-squared value.

    Formula:
        R^2 = 1 - (sum((y_actual - y_pred)^2) / sum((y_actual - mean(y_actual))^2))
    """
    if len(y_pred) != len(y_actual):
        raise ValueError("The lengths of y_pred and y_actual do not match.")

    ssr = np.sum((y_actual - y_pred) ** 2)
    sst = np.sum((y_actual - np.mean(y_actual)) ** 2)
    r2 = 1 - (ssr / sst)
    return r2


def adjusted_r_squared(y_pred, y_actual, n_features):
    """
    Calculate the Adjusted R-squared for regression.

    Args:
        y_pred (array-like): Predicted target values.
        y_actual (array-like): Actual target values.
        n_features (int): Number of features in the model.

    Returns:
        float: Adjusted R-squared value.

    Formula:
        Adjusted R^2 = 1 - ((1 - R^2) * (n - 1) / (n - k - 1))
    """
    if len(y_pred) != len(y_actual):
        raise ValueError("The lengths of y_pred and y_actual do not match.")

    r2 = r_squared(y_pred, y_actual)
    n = len(y_actual)
    k = n_features
    adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - k - 1))
    return adj_r2


def mean_squared_log_error(y_pred, y_actual):
    """
    Calculate the Mean Squared Logarithmic Error (MSLE) for regression.

    Args:
        y_pred (array-like): Predicted target values.
        y_actual (array-like): Actual target values.

    Returns:
        float: Mean Squared Logarithmic Error.

    Formula:
        MSLE = (1/n) * sum(log(1 + y_pred) - log(1 + y_actual))^2)
    """
    if len(y_pred) != len(y_actual):
        raise ValueError("The lengths of y_pred and y_actual do not match.")

    msle = np.mean((np.log1p(y_pred) - np.log1p(y_actual)) ** 2)
    return msle


def root_mean_squared_log_error(y_pred, y_actual):
    """
    Calculate the Root Mean Squared Logarithmic Error (RMSLE) for regression.

    Args:
        y_pred (array-like): Predicted target values.
        y_actual (array-like): Actual target values.

    Returns:
        float: Root Mean Squared Logarithmic Error.

    Formula:
        RMSLE = sqrt(MSLE)
    """
    if len(y_pred) != len(y_actual):
        raise ValueError("The lengths of y_pred and y_actual do not match.")

    msle = np.mean((np.log1p(y_pred) - np.log1p(y_actual)) ** 2)
    rmsle = np.sqrt(msle)
    return rmsle


def mean_percentage_error(y_pred, y_actual):
    """
    Calculate the Mean Percentage Error (MPE) for regression.

    Args:
        y_pred (array-like): Predicted target values.
        y_actual (array-like): Actual target values.

    Returns:
        float: Mean Percentage Error.

    Formula:
        MPE = (1/n) * sum((y_actual - y_pred) / y_actual) * 100
    """
    if len(y_pred) != len(y_actual):
        raise ValueError("The lengths of y_pred and y_actual do not match.")

    mpe = np.mean((y_actual - y_pred) / y_actual) * 100
    return mpe


def mean_absolute_percentage_error(y_pred, y_actual):
    """
    Calculate the Mean Absolute Percentage Error (MAPE) for regression.

    Args:
        y_pred (array-like): Predicted target values.
        y_actual (array-like): Actual target values.

    Returns:
        float: Mean Absolute Percentage Error.

    Formula:
        MAPE = (1/n) * sum(abs((y_actual - y_pred) / y_actual)) * 100
    """
    if len(y_pred) != len(y_actual):
        raise ValueError("The lengths of y_pred and y_actual do not match.")

    mape = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100
    return mape


def median_absolute_error(y_pred, y_actual):
    """
    Calculate the Median Absolute Error (MedAE) for regression.

    Args:
        y_pred (array-like): Predicted target values.
        y_actual (array-like): Actual target values.

    Returns:
        float: Median Absolute Error.
    """
    if len(y_pred) != len(y_actual):
        raise ValueError("The lengths of y_pred and y_actual do not match.")

    medae = np.median(np.abs(y_pred - y_actual))
    return medae
