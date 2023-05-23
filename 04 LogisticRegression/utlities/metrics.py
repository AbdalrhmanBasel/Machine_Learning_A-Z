import numpy as np


def accuracy(y_true, y_pred):
    """
    Computes the accuracy of the model's predictions.
    :param y_true: True labels
    :param y_pred: Predicted labels
    :return: Accuracy score
    """
    return np.mean(y_true == y_pred)


def precision(y_true, y_pred):
    """
    Computes the precision of the model's predictions.
    :param y_true: True labels
    :param y_pred: Predicted labels
    :return: Precision score
    """
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    false_positives = np.sum((y_true == 0) & (y_pred == 1))
    precision = true_positives / (true_positives + false_positives + 1e-8)
    return precision


def recall(y_true, y_pred):
    """
    Computes the recall of the model's predictions.
    :param y_true: True labels
    :param y_pred: Predicted labels
    :return: Recall score
    """
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    false_negatives = np.sum((y_true == 1) & (y_pred == 0))
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    return recall


def f1_score(y_true, y_pred):
    """
    Computes the F1 score of the model's predictions.
    :param y_true: True labels
    :param y_pred: Predicted labels
    :return: F1 score
    """
    precision_val = precision(y_true, y_pred)
    recall_val = recall(y_true, y_pred)
    f1_score = 2 * (precision_val * recall_val) / (precision_val + recall_val + 1e-8)
    return f1_score
