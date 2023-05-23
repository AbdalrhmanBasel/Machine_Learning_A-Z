import numpy as np


# Step 2: Dataset Splitting
def train_test_validation_split(X, y, test_size=0.2, val_size=0.2):
    # Get the total number of examples in the dataset
    num_examples = X.shape[0]

    # Calculate the number of examples for the test and validation sets
    num_test = int(test_size * num_examples)
    num_val = int(val_size * num_examples)

    # Generate a random permutation of indices for shuffling the data
    indices = np.random.permutation(num_examples)

    # Split the indices into test, validation, and training sets
    test_indices = indices[:num_test]
    val_indices = indices[num_test:num_test + num_val]
    train_indices = indices[num_test + num_val:]

    # Split the data based on the computed indices
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    X_val = X[val_indices]
    y_val = y[val_indices]

    # Return the split data
    return X_train, X_test, X_val, y_train, y_test, y_val
