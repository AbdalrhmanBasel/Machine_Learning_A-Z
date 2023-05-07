import numpy as np
from easydict import EasyDict


class PolynomialRegression:
    def __init__(self, cfg: EasyDict):
        self.weights = np.random.randn(cfg.degree + 1)
        self.degree = cfg.degree
        self.reg_coeff = cfg.reg_coeff
        self.learning_rate = cfg.learning_rate
        self.n_iterations = cfg.n_iterations

    def __design_matrix(self, inputs: np.ndarray) -> np.ndarray:
        """
        Build the design matrix using the polynomial basis functions.
        """
        n_samples = inputs.shape[0]
        design_matrix = np.zeros((n_samples, self.degree + 1))
        for i in range(self.degree + 1):
            design_matrix[:, i] = inputs ** i
        return design_matrix

    def __calculate_weights(self, design_matrix: np.ndarray, targets: np.ndarray) -> None:
        """
        Calculate the model weights using gradient descent.
        """
        self.weights = np.zeros(self.degree + 1)
        n_samples = design_matrix.shape[0]

        for i in range(self.n_iterations):
            # Calculate the predictions and the error
            predictions = design_matrix.dot(self.weights)
            error = predictions - targets

            # Calculate the gradient and update the weights
            gradient = 2 / n_samples * design_matrix.T.dot(error) + 2 * self.reg_coeff * self.weights
            self.weights -= self.learning_rate * gradient

    def fit(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        """
        Train the polynomial regression model using the input data and target values.
        """
        # Construct the design matrix using the input data
        design_matrix = self.__design_matrix(inputs)

        # Calculate the model weights using gradient descent
        self.__calculate_weights(design_matrix, targets)

    def calculate_model_prediction(self, design_matrix) -> np.ndarray:
        """
        Polynomial Regression formula
        y = a0 + a1*x + a2*x^2 + ... + an*x^n
        """
        return design_matrix.dot(self.weights)

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Use the trained model to make predictions on new input data.
        """
        design_matrix = self.__design_matrix(inputs)
        predictions = self.calculate_model_prediction(design_matrix)
        return predictions
