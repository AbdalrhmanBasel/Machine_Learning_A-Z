import easydict
import numpy as np
from matplotlib import pyplot as plt

from configs.cfg import cfg


class PlayerClassifier:
    def __init__(self, cfg:easydict):
        """
        Constructor for the SportsmanClassifier class.
        """
        self.threshold = cfg.threshold
        self.max_height = cfg.max_height

    def calculate_confidences(self, height):
        """
        Takes height and returns confidence of belonging to
        the class of basketball players
        """
        confidence = height / self.max_height
        return confidence

    def classify(self, confidences):
        """
        This function takes confidence and threshold and returns a classification predication.
        """
        classification_list = []

        for confidence in confidences:
            if confidence >= self.threshold:
                classification_list.append(1)  # Basketball player
            else:
                classification_list.append(0)  # Football player

        return np.array(classification_list)

    def calculate_metrics(self, y_true, y_pred):
        tp = np.sum(np.logical_and(y_true == 1, y_pred == 1))
        tn = np.sum(np.logical_and(y_true == 0, y_pred == 0))
        fp = np.sum(np.logical_and(y_true == 0, y_pred == 1))
        fn = np.sum(np.logical_and(y_true == 1, y_pred == 0))

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * precision * recall / (precision + recall)

        return accuracy, precision, recall, f1_score

    def evaluate(self, X, y, threshold):
        # Use the trained classifier to obtain predicted probabilities
        y_pred_prob = self.classify(np.column_stack((np.ones(X.shape[0]), X)))

        # Compute precision and recall for different classification thresholds
        thresholds = np.linspace(0, 1, num=100)
        precisions = []
        recalls = []

        for threshold in thresholds:
            y_pred = np.where(y_pred_prob >= threshold, 1, 0)
            true_positives = np.sum((y == 1) & (y_pred == 1))
            false_positives = np.sum((y == 0) & (y_pred == 1))
            false_negatives = np.sum((y == 1) & (y_pred == 0))
            precision = true_positives / (true_positives + false_positives)
            recall = true_positives / (true_positives + false_negatives)
            precisions.append(precision)
            recalls.append(recall)

        # Plot Precision-Recall Curve
        plt.figure(figsize=(8, 6))
        plt.plot(recalls, precisions)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.show()

        # Choose Classification Threshold
        # Choose an appropriate classification threshold based on the specific application requirements

        # Evaluate on Test Set
        # Evaluate the trained classifier using the test set to obtain performance metrics
        y_pred = np.where(y_pred_prob >= threshold, 1, 0)  # Choose classification threshold of 0.5
        true_positives = np.sum((y == 1) & (y_pred == 1))
        false_positives = np.sum((y == 0) & (y_pred == 1))
        false_negatives = np.sum((y == 1) & (y_pred == 0))
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1_score = 2 * (precision * recall) / (precision + recall)

        print("Precision: {:.4f}".format(precision))
        print("Recall: {:.4f}".format(recall))
        print("F1-Score: {:.4f}".format(f1_score))



