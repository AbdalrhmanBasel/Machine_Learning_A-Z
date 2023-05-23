import numpy as np


def mean_squared_error(y_true, y_pred):
    """
    Calculates the mean squared error (MSE) loss.

    MSE = (1/n) * sum((y_true - y_pred)^2)

    Args:
        y_true (numpy.ndarray): Array of true values.
        y_pred (numpy.ndarray): Array of predicted values.

    Returns:
        float: Mean squared error loss.
    """
    n = len(y_true)
    mse = np.mean((y_true - y_pred) ** 2)
    return mse


def mean_absolute_error(y_true, y_pred):
    """
    Calculates the mean absolute error (MAE) loss.

    MAE = (1/n) * sum(|y_true - y_pred|)

    Args:
        y_true (numpy.ndarray): Array of true values.
        y_pred (numpy.ndarray): Array of predicted values.

    Returns:
        float: Mean absolute error loss.
    """
    n = len(y_true)
    mae = np.mean(np.abs(y_true - y_pred))
    return mae


def root_mean_squared_error(y_true, y_pred):
    """
    Calculates the root mean squared error (RMSE) loss.

    RMSE = sqrt((1/n) * sum((y_true - y_pred)^2))

    Args:
        y_true (numpy.ndarray): Array of true values.
        y_pred (numpy.ndarray): Array of predicted values.

    Returns:
        float: Root mean squared error loss.
    """
    n = len(y_true)
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    return rmse


def r_squared(y_true, y_pred):
    """
    Calculates the R-squared (coefficient of determination).

    R^2 = 1 - (sum((y_true - y_pred)^2) / sum((y_true - y_mean)^2))

    Args:
        y_true (numpy.ndarray): Array of true values.
        y_pred (numpy.ndarray): Array of predicted values.

    Returns:
        float: R-squared value.
    """
    y_mean = np.mean(y_true)
    ss_total = np.sum((y_true - y_mean) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    return r2


def mean_squared_logarithmic_error(y_true, y_pred):
    """
    Calculates the mean squared logarithmic error (MSLE) loss.

    MSLE = (1/n) * sum((log(1 + y_true) - log(1 + y_pred))^2)

    Args:
        y_true (numpy.ndarray): Array of true values.
        y_pred (numpy.ndarray): Array of predicted values.

    Returns:
        float: Mean squared logarithmic error loss.
    """
    n = len(y_true)
    msle = np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)
    return msle


def huber_loss(y_true, y_pred, delta=1.0):
    """
    Calculates the Huber loss.

    Huber Loss = (1/n) * sum(huber_loss_per_sample)

    where,
    huber_loss_per_sample = 0.5 * (y_true - y_pred)^2                  if |y_true - y_pred| <= delta
                           delta * |y_true - y_pred| - 0.5 * delta^2  otherwise

    Args:
        y_true (numpy.ndarray): Array of true values.
        y_pred (numpy.ndarray): Array of predicted values.
        delta (float): Threshold value.

    Returns:
        float: Huber loss.
    """
    n = len(y_true)
    residuals = y_true - y_pred
    huber_loss_per_sample = np.where(np.abs(residuals) <= delta, 0.5 * residuals ** 2,
                                     delta * np.abs(residuals) - 0.5 * delta ** 2)
    huber_loss = np.mean(huber_loss_per_sample)
    return huber_loss


def quantile_loss(y_true, y_pred, q):
    """
    Calculates the quantile loss.

    Quantile Loss = (1/n) * sum(quantile_loss_per_sample)

    where,
    quantile_loss_per_sample = (q * (y_true - y_pred))    if y_true >= y_pred
                               ((1 - q) * (y_pred - y_true))    otherwise

    Args:
        y_true (numpy.ndarray): Array of true values.
        y_pred (numpy.ndarray): Array of predicted values.
        q (float): Quantile value.

    Returns:
        float: Quantile loss.
    """
    n = len(y_true)
    quantile_loss_per_sample = np.where(y_true >= y_pred, q * (y_true - y_pred), (1 - q) * (y_pred - y_true))
    quantile_loss = np.mean(quantile_loss_per_sample)
    return quantile_loss


def log_cosh_loss(y_true, y_pred):
    """
    Calculates the log-cosh loss.

    Log-Cosh Loss = (1/n) * sum(log(cosh(y_true - y_pred)))

    Args:
        y_true (numpy.ndarray): Array of true values.
        y_pred (numpy.ndarray): Array of predicted values.

    Returns:
        float: Log-cosh loss.
    """
    n = len(y_true)
    log_cosh_loss = np.mean(np.log(np.cosh(y_true - y_pred)))
    return log_cosh_loss
