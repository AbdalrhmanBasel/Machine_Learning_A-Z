import numpy as np


def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Split the data into training and testing sets.

    Parameters:
    - X: Input features
    - y: Target labels
    - test_size: The fraction of the data to be used for testing (default: 0.2)
    - random_state: Seed for the random number generator (optional)

    Returns:
    - X_train: Training features
    - X_test: Testing features
    - y_train: Training labels
    - y_test: Testing labels
    """

    # Check if the size is valid
    if test_size < 0 or test_size > 1:
        raise ValueError("Invalid test_size: must be between 0 and 1.")

    # Set the random seed if provided
    if random_state is not None:
        np.random.seed(random_state)

    # Shuffle the data
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    # Calculate the size of the test set
    test_split = int(len(X) * test_size)

    # Split the data
    X_train = X[test_split:]
    X_test = X[:test_split]

    y_train = y[test_split:]
    y_test = y[:test_split]

    return X_train, X_test, y_train, y_test


def train_test_val_split(X, y, test_size=0.2, val_size=0.2, random_state=None):
    """
    Split the data into training, testing, and validation sets.

    Parameters:
    - X: Input features
    - y: Target labels
    - test_size: The fraction of the data to be used for testing (default: 0.2)
    - val_size: The fraction of the data to be used for validation (default: 0.2)
    - random_state: Seed for the random number generator (optional)

    Returns:
    - X_train: Training features
    - X_test: Testing features
    - X_val: Validation features
    - y_train: Training labels
    - y_test: Testing labels
    - y_val: Validation labels

    TO USE ===>
        X_train, X_test, X_val, y_train, y_test, y_val = train_test_val_split(X, y, test_size=0.2, val_size=0.2, random_state=42)
    """

    # Check if the sizes are valid
    if test_size < 0 or test_size > 1:
        raise ValueError("Invalid test_size: must be between 0 and 1.")
    if val_size < 0 or val_size > 1:
        raise ValueError("Invalid val_size: must be between 0 and 1.")
    if test_size + val_size >= 1:
        raise ValueError("Invalid test_size and val_size: their sum must be less than 1.")

    # Set the random seed if provided
    if random_state is not None:
        np.random.seed(random_state)

    # Shuffle the data
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    # Calculate the sizes of each set
    test_split = int(len(X) * test_size)
    val_split = int(len(X) * val_size)

    # Split the data
    X_test = X[:test_split]
    X_val = X[test_split:test_split + val_split]
    X_train = X[test_split + val_split:]

    y_test = y[:test_split]
    y_val = y[test_split:test_split + val_split]
    y_train = y[test_split + val_split:]

    return X_train, X_test, X_val, y_train, y_test, y_val
