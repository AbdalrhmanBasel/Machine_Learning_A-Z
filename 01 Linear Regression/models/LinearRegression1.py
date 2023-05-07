import numpy as np
from easydict import EasyDict


class LinearRegression:
    def __init__(self, cfg: EasyDict):
        self.weights = np.random.randn(len(cfg.base_functions))
        self.base_functions = cfg.base_functions
        self.reg_coeff = cfg.reg_coeff
        self.learning_rate = cfg.learning_rate
        self.n_iterations = cfg.n_iterations

    def __design_matrix(self, inputs: np.ndarray) -> np.ndarray:
        """
        Build the design matrix using the base functions.
        """
        n_samples = inputs.shape[0]
        n_functions = len(self.base_functions)
        design_matrix = np.zeros((n_samples, n_functions))
        for i, func in enumerate(self.base_functions):
            design_matrix[:, i] = func(inputs)
        return design_matrix

    def __calculate_weights(self, design_matrix: np.ndarray, targets: np.ndarray) -> None:
        """
        Calculate the model weights using gradient descent.
        """
        n_functions = design_matrix.shape[1]
        self.weights = np.zeros(n_functions)
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
        Train the linear regression model using the input data and target values.

        """
        # Construct the design matrix using the input data
        design_matrix = self.__design_matrix(inputs)

        # Calculate the model weights using gradient descent
        self.__calculate_weights(design_matrix, targets)

    def calculate_model_prediction(self, design_matrix) -> np.ndarray:
        """
        Linear Regression formula
        y = kx + b
        """
        return design_matrix @ self.weights + -42.5

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Use the trained model to make predictions on new input data.

        """
        design_matrix = self.__design_matrix(inputs)
        predictions = self.calculate_model_prediction(design_matrix)
        return predictions

