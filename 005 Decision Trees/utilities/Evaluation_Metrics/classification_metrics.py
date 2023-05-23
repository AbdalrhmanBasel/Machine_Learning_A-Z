import numpy as np

def accuracy(y_pred, y_actual):
    """
    Calculate the accuracy of a classification model.

    Args:
        y_pred (list or numpy array): Predicted target values.
        y_actual (list or numpy array): Actual target values.

    Returns:
        float: Accuracy score between 0 and 1.

    Formula:
        Accuracy = (TP + TN) / (TP + TN + FP + FN)
    """
    # Check if the predicted and actual values have the same length
    if len(y_pred) != len(y_actual):
        raise ValueError("The length of y_pred is not equal to y_actual. Check if their sizes are equal.")

    # Calculate the number of correctly classified instances
    correct_count = sum(1 for pred, actual in zip(y_pred, y_actual) if pred == actual)

    # Calculate the total number of instances
    total_count = len(y_pred)

    # Calculate accuracy
    accuracy_score = correct_count / total_count

    return accuracy_score


def precision(y_pred, y_actual):
    """
    Calculate the precision of a classification model.

    Args:
        y_pred (list or numpy array): Predicted target values.
        y_actual (list or numpy array): Actual target values.

    Returns:
        float: Precision score between 0 and 1.

    Formula:
        Precision = TP / (TP + FP)
    """
    # Check if the predicted and actual values have the same length
    if len(y_pred) != len(y_actual):
        raise ValueError("The lengths of y_pred and y_actual do not match.")

    # Calculate the number of true positive predictions
    true_positives = sum(1 for pred, actual in zip(y_pred, y_actual) if pred == 1 and actual == 1)

    # Calculate the number of positive predictions
    positive_predictions = sum(1 for pred in y_pred if pred == 1)

    # Calculate precision
    precision_score = true_positives / positive_predictions if positive_predictions != 0 else 0

    return precision_score


def recall(y_pred, y_actual):
    """
    Calculate the recall (sensitivity) of a classification model.

    Args:
        y_pred (list or numpy array): Predicted target values.
        y_actual (list or numpy array): Actual target values.

    Returns:
        float: Recall score between 0 and 1.

    Formula:
        Recall = TP / (TP + FN)
    """
    # Check if the predicted and actual values have the same length
    if len(y_pred) != len(y_actual):
        raise ValueError("The lengths of y_pred and y_actual do not match.")

    # Calculate the number of true positive predictions
    true_positives = sum(1 for pred, actual in zip(y_pred, y_actual) if pred == 1 and actual == 1)

    # Calculate the number of actual positive instances
    actual_positives = sum(1 for actual in y_actual if actual == 1)

    # Calculate recall
    recall_score = true_positives / actual_positives if actual_positives != 0 else 0

    return recall_score


def f1_score(y_pred, y_actual):
    """
    Calculate the F1-score of a classification model.

    Args:
        y_pred (list or numpy array): Predicted target values.
        y_actual (list or numpy array): Actual target values.

    Returns:
        float: F1-score between 0 and 1.

    Formula:
        F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
    """
    # Check if the predicted and actual values have the same length
    if len(y_pred) != len(y_actual):
        raise ValueError("The lengths of y_pred and y_actual do not match.")

    # Calculate precision and recall using the previously defined functions
    precision_score = precision(y_pred, y_actual)
    recall_score = recall(y_pred, y_actual)

    # Calculate F1-score
    f1_score = 2 * (precision_score * recall_score) / (precision_score + recall_score) \
        if (precision_score + recall_score) != 0 else 0

    return f1_score


def auc_roc(y_pred, y_actual):
    """
    Calculate the Area Under the ROC Curve (AUC-ROC) of a classification model.

    Args:
        y_pred (list or numpy array): Predicted target probabilities.
        y_actual (list or numpy array): Actual target values.

    Returns:
        float: AUC-ROC score between 0 and 1.
    """
    # Check if the predicted and actual values have the same length
    if len(y_pred) != len(y_actual):
        raise ValueError("The lengths of y_pred and y_actual do not match.")

    # Sort the predicted probabilities and actual values by descending order of probabilities
    sorted_indices = np.argsort(y_pred)[::-1]
    y_pred_sorted = y_pred[sorted_indices]
    y_actual_sorted = y_actual[sorted_indices]

    # Calculate the total number of positive instances
    total_positives = np.sum(y_actual)

    # Calculate the true positive rate (TPR) and false positive rate (FPR) for various thresholds
    tpr = np.cumsum(y_actual_sorted) / total_positives
    fpr = np.cumsum(1 - y_actual_sorted) / (len(y_actual) - total_positives)

    # Calculate the AUC-ROC score using the trapezoidal rule
    auc_roc_score = np.trapz(tpr, fpr)

    return auc_roc_score


def log_loss(y_pred, y_actual):
    """
    Calculate the Log Loss (Cross-Entropy Loss) of a classification model.

    Args:
        y_pred (list or numpy array): Predicted target probabilities.
        y_actual (list or numpy array): Actual target values.

    Returns:
        float: Log Loss score.

    Formula:
        Log Loss = -1/n * Σ(y * log(y_pred) + (1 - y) * log(1 - y_pred)),
        where y is the actual target value and y_pred is the predicted probability.
    """
    # Check if the predicted and actual values have the same length
    if len(y_pred) != len(y_actual):
        raise ValueError("The lengths of y_pred and y_actual do not match.")

    # Calculate the Log Loss
    log_loss_score = -1 / len(y_actual) * np.sum(y_actual * np.log(y_pred) + (1 - y_actual) * np.log(1 - y_pred))

    return log_loss_score


def cohen_kappa(y_pred, y_actual):
    """
    Calculate Cohen's Kappa of a classification model.

    Args:
        y_pred (list or numpy array): Predicted target values.
        y_actual (list or numpy array): Actual target values.

    Returns:
        float: Cohen's Kappa score between -1 and 1.
    """
    # Check if the predicted and actual values have the same length
    if len(y_pred) != len(y_actual):
        raise ValueError("The lengths of y_pred and y_actual do not match.")

    # Calculate the observed agreement
    total_count = len(y_pred)
    observed_agreement = sum(1 for pred, actual in zip(y_pred, y_actual) if pred == actual) / total_count

    # Calculate the expected agreement
    expected_agreement = sum(np.sum(y_pred == label) * np.sum(y_actual == label) for label in set(y_actual)) / total_count ** 2

    # Calculate Cohen's Kappa
    kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)

    return kappa


def confusion_matrix(y_pred, y_actual):
    """
    Calculate the confusion matrix of a classification model.

    Args:
        y_pred (list or numpy array): Predicted target values.
        y_actual (list or numpy array): Actual target values.

    Returns:
        numpy array: Confusion matrix.
    """
    # Check if the predicted and actual values have the same length
    if len(y_pred) != len(y_actual):
        raise ValueError("The lengths of y_pred and y_actual do not match.")

    # Create an empty confusion matrix
    classes = np.unique(np.concatenate((y_pred, y_actual)))
    num_classes = len(classes)
    cm = np.zeros((num_classes, num_classes))

    # Fill the confusion matrix
    for pred, actual in zip(y_pred, y_actual):
        cm[pred][actual] += 1

    return cm
