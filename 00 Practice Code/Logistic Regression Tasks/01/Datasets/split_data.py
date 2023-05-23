import numpy as np


def train_test_split(X, y, test_size=0.3, random_state=None):
    """
    Split the dataset into training and testing sets with a given test size ratio.

    Parameters:
        X (numpy array): Feature matrix
        y (numpy array): Target vector
        test_size (float): Ratio of the testing set size (default: 0.3)
        random_state (int): Random seed (default: None)

    Returns:
        X_train (numpy array): Feature matrix of the training set
        X_test (numpy array): Feature matrix of the testing set
        y_train (numpy array): Target vector of the training set
        y_test (numpy array): Target vector of the testing set
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Shuffle the indices
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    # Split the indices
    n_test = int(test_size * X.shape[0])
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    # Split the dataset
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test
