import numpy as np


class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)

    def transform(self, X):
        if self.std_ is not None and np.any(self.std_ != 0):
            transformed_X = (X - self.mean_) / self.std_
            transformed_X[np.isnan(transformed_X)] = 0  # Replace NaN values with zeros
            return transformed_X
        else:
            return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
