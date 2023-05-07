import numpy as np


def MSE(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Function that calculates the loss function (mean squared error)
    and returns the difference between the actual values and predicted values.
    """
    mse = np.mean((targets - predictions) ** 2)
    return mse
