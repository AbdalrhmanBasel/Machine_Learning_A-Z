import numpy as np
from datasets.Standarization import StandardScaler
from sklearn.utils import resample


class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=5000, penalty=None, C=1.0, class_weight=None):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.penalty = penalty
        self.C = C
        self.class_weight = class_weight
        self.weights = None
        self.bias = None

    def design_matrix(self, X):
        X = np.asarray(X)
        if len(X.shape) == 1:
            X = X[:, np.newaxis]
        ones = np.ones((X.shape[0], 1))
        return np.concatenate((ones, X), axis=1)

    def fit(self, X, y):
        self.X = self.design_matrix(X)
        self.m, self.n = self.X.shape
        self.weights = np.zeros(self.n)
        self.bias = 0
        self.y = y

        # Perform feature scaling
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)

        # Handle class imbalance
        X_resampled, y_resampled = resample(self.X, self.y, stratify=self.y)

        for _ in range(self.n_iterations):
            self._update_weights(X_resampled, y_resampled)

    def _update_weights(self, X, y):
        z = np.dot(X, self.weights) + self.bias
        y_hat = self.sigmoid(z)

        dw = (1 / self.m) * np.dot(X.T, (y_hat - y))
        db = (1 / self.m) * np.sum(y_hat - y)

        if self.penalty is not None:
            dw += self.penalty_derivative()

        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db

    def penalty_derivative(self):
        if self.penalty == 'l1':
            return self.C * np.sign(self.weights)
        elif self.penalty == 'l2':
            return 2 * self.C * self.weights
        else:
            return 0

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def predict(self, X):
        X = self.design_matrix(X)
        X = StandardScaler().fit_transform(X)  # Scale the input features
        z = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(z)
        y_pred = np.where(y_pred > 0.5, 1, 0)
        return y_pred
