import numpy as np


def cross_entropy_loss(y_true, y_pred):
    """
    Compute the cross-entropy loss between true class labels and predicted class probabilities.

    Args:
        y_true (ndarray): Array of true class labels (shape: [n_samples]).
        y_pred (ndarray): Array of predicted class probabilities (shape: [n_samples, n_classes]).

    Returns:
        float: Cross-entropy loss.
    """
    epsilon = 1e-10  # small value to avoid division by zero
    n_samples = len(y_true)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # clip predicted probabilities to avoid log(0)
    loss = -np.sum(y_true * np.log(y_pred)) / n_samples
    return loss


def hinge_loss(y_true, y_pred):
    """
    Compute the hinge loss between true class labels and predicted class scores.

    Args:
        y_true (ndarray): Array of true class labels (-1 or 1) (shape: [n_samples]).
        y_pred (ndarray): Array of predicted class scores (shape: [n_samples]).

    Returns:
        float: Hinge loss.
    """
    n_samples = len(y_true)
    loss = np.mean(np.maximum(0, 1 - y_true * y_pred))
    return loss


def log_loss(y_true, y_pred):
    """
    Compute the logarithmic loss (log loss) between true binary labels and predicted class probabilities.

    Args:
        y_true (ndarray): Array of true binary labels (0 or 1) (shape: [n_samples]).
        y_pred (ndarray): Array of predicted class probabilities (shape: [n_samples]).

    Returns:
        float: Logarithmic loss.
    """
    epsilon = 1e-10  # small value to avoid division by zero
    n_samples = len(y_true)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # clip predicted probabilities to avoid log(0)
    loss = -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) / n_samples
    return loss


def exponential_loss(y_true, y_pred):
    """
    Compute the exponential loss (AdaBoost loss) between true class labels and predicted class scores.

    Args:
        y_true (ndarray): Array of true class labels (-1 or 1) (shape: [n_samples]).
        y_pred (ndarray): Array of predicted class scores (shape: [n_samples]).

    Returns:
        float: Exponential loss.
    """
    n_samples = len(y_true)
    loss = np.mean(np.exp(-y_true * y_pred))
    return loss


def sigmoid_loss(y_true, y_pred):
    """
    Compute the sigmoid loss between true binary labels and predicted class probabilities.

    Args:
        y_true (ndarray): Array of true binary labels (0 or 1) (shape: [n_samples]).
        y_pred (ndarray): Array of predicted class probabilities (shape: [n_samples]).

    Returns:
        float: Sigmoid loss.
    """
    n_samples = len(y_true)
    loss = np.mean(np.log1p(np.exp(-y_true * y_pred)))
    return loss


def kullback_leibler_divergence(p, q):
    """
    Compute the Kullback-Leibler (KL) divergence between two probability distributions p and q.

    Args:
        p (ndarray): Array representing the true probability distribution (shape: [n_samples]).
        q (ndarray): Array representing the predicted probability distribution (shape: [n_samples]).

    Returns:
        float: KL divergence.
    """
    epsilon = 1e-10  # small value to avoid division by zero
    p = np.clip(p, epsilon, 1)  # clip true probabilities to avoid log(0)
    q = np.clip(q, epsilon, 1)  # clip predicted probabilities to avoid log(0)
    loss = np.sum(p * np.log(p / q))
    return loss
