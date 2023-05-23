import numpy as np


def split_data(X, y, train_size=0.8, valid_size=0.1, test_size=0.1, shuffle=True, random_state=None):
    """
    Split a dataset into training, validation, and testing sets for classification.
    """

    # Check if the sum of train_size, valid_size, and test_size is equal to 1.0.
    assert train_size + valid_size + test_size == 1.0, "The sum of train_size, val_size, and test_size should be 1.0."

    # Get the number of samples in the dataset.
    n_samples = len(X)

    # Get the indices of the samples.
    indices = np.arange(n_samples)


    # Shuffle the indices if shuffle is True.
    if shuffle:
        if random_state:
            np.random.seed(random_state)
        np.random.shuffle(indices)

    # Get the end indices of the training, validation, and testing sets.
    train_end = int(n_samples * train_size)
    val_end = int(n_samples * (train_size + valid_size))

    # Get the indices of the samples for the training, validation, and testing sets.
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    # Get the features and labels for the training, validation, and testing sets.
    train_features, train_labels = X[train_indices], y[train_indices]
    val_features, val_labels = X[val_indices], y[val_indices]
    test_features, test_labels = X[test_indices], y[test_indices]

    # Return the features and labels for the training, validation, and testing sets.
    return train_features, train_labels, val_features, val_labels, test_features, test_labels
