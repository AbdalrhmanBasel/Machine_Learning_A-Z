import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=10000, fit_intercept=True, verbose=False):
        """
        The LogisticRegression class constructor takes the following arguments:
        - learning_rate (float): The learning rate to be used in the gradient descent algorithm (default: 0.01).
        - num_iterations (int): The number of iterations to be used in the gradient descent algorithm (default: 100000).
        - fit_intercept (bool): Whether or not to include an intercept term in the model (default: True).
        - verbose (bool): Whether or not to print the cost function value at every 10000 iterations (default: False).
        """
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.fit_intercept = fit_intercept
        self.verbose = verbose

    def add_intercept(self, X):
        """
        This method adds an intercept term to the input feature matrix X.
        """
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def sigmoid(self, z):
        """
        This method applies the sigmoid function to the input argument z.
        """
        return 1 / (1 + np.exp(-z))

    def cost_function(self, X, y, weights):
        """
        This method calculates the cost function (calculate accuracy) of the logistic regression model
        using a cost function called binary cross-entropy.

        Function:
        J(w) = - (1/m) * [ y*log(h) + (1-y)*log(1-h) ]

        where:
        J(w) is the cost function to be minimized.
        w is a vector of the model's parameters (i.e., weights and bias).
        m is the number of training examples.
        y is a vector of the true labels (either 0 or 1) for each training example.
        h is a vector of the predicted probabilities (i.e., the output of the sigmoid function) for each training example.
        """
        # l_Regression =z = X @ weights = X[0] * w[0] + X[1] * w[1] + ... + X[n] * w[n]
        z = X @ weights
        h = self.sigmoid(z)
        cost = (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
        return cost

    def fit(self, X, y):
        """
        This method trains the logistic regression model using the input feature matrix X and target vector y.
        """
        if self.fit_intercept:
            X = self.add_intercept(X)

        # initialize weights to zeros
        self.weights = np.zeros(X.shape[1])

        # Gradient Descent
        for i in range(self.num_iterations):
            z = X @ self.weights  # Linear Regression Prediction Formula
            h = self.sigmoid(z)  # convert predicted values into to a probability score.
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