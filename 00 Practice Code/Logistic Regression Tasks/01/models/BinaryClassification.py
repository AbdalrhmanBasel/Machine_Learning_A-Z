import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate=0.1, n_iterations=100000, fit_intercept=True):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.fit_intercept = fit_intercept

    def add_intercept(self, X):
        """
        This method adds an intercept term to the input feature matrix X.
        """
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def sigmoid(self, z):
        """
        :param z:is the linear function z = w0 + w1x1 + w2x2 + ... + wn*xn
        :return: the sigmoid function of z.
        """
        z = 1 / (1 + np.exp(-z))
        return z

    def cost_function(self, X, y, weights):
        """
        This function calculates the cross-entropy of binary classification function
        to measure the performance of the binary model.
        """
        z = X @ weights
        h = self.sigmoid(z)
        cross_entropy = -(y * np.log(h) + (1 - y) * np.log(1 - h)).mean()
        return cross_entropy

    def fit(self, X, y):

        if self.fit_intercept:
            X = self.add_intercept(X)

        self.weights = np.zeros(X.shape[1])

        for i in range(self.n_iterations):
            z = X @ self.weights  # Linear Regression prediction
            h = self.sigmoid(z)  # Converting valuaes to probablities from 0 to 1
            gradient = X.T @ (h - y) / y.size
            self.weights -= self.learning_rate * gradient

            if self.verbose and i % 1000 == 0:
                print(f"Cost at iteration {i}: {self.cost_function(X, y, self.weights)}")

    def predict_probabilities(self, X):
        """
        This method predicts the probabilities of the target variable using the trained logistic regression model.
        """
        if self.fit_intercept:
            X = self.add_intercept(X)

        return self.sigmoid(X @ self.weights)

    def predict(self, X, threshold=0.5):
        """
        This method predicts the binary target variable based on a threshold (default: 0.5).
        """
        return self.predict_probabilities(X) >= threshold


