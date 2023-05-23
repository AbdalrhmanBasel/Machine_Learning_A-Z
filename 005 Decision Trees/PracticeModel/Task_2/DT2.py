import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class Node:
    """
    Node class to represent a node in the decision tree.
    """

    def __init__(self, left=None, right=None, feature_index=None, threshold=None, predicted_value=None):
        self.left = left  # Left subtree
        self.right = right  # Right subtree
        self.feature_index = feature_index  # Index of the feature to split on
        self.threshold = threshold  # Threshold value for splitting
        self.predicted_value = predicted_value  # Predicted value at the leaf node


class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, random_state=None):
        self.max_depth = max_depth  # Maximum depth of the decision tree
        self.min_samples_split = min_samples_split  # Minimum number of samples required to perform a split
        self.random_state = random_state  # Random seed for reproducibility
        self.root = None  # Root node of the decision tree

    def fit(self, X, y):
        """
        Fit the decision tree to the training data.
        """
        np.random.seed(self.random_state)  # Set the random seed
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        """
        Recursively build the decision tree by splitting the data based on the best feature and threshold.
        """
        if depth >= self.max_depth or len(X) < self.min_samples_split or len(np.unique(y)) == 1:
            return Node(predicted_value=self._calculate_leaf_value(y))  # Create a leaf node and assign the predicted value

        n_features = X.shape[1]  # Number of features in the input dataset X
        best_feature, best_threshold = None, None
        best_info_gain = -np.inf  # Initialize the best information gain as negative infinity

        # Randomly select a subset of features
        random_features = np.random.choice(n_features, size=int(np.sqrt(n_features)), replace=False)

        # Iterate over each random feature and threshold to find the best split
        for feature_index in random_features:
            unique_values = np.unique(X[:, feature_index])
            threshold = np.random.choice(unique_values)  # Randomly select a threshold
            info_gain = self._calculate_info_gain(X, y, feature_index, threshold)
            if info_gain > best_info_gain:
                best_feature = feature_index
                best_threshold = threshold
                best_info_gain = info_gain

        # Split the data based on the best feature and threshold
        left_indices = X[:, best_feature] <= best_threshold
        right_indices = X[:, best_feature] > best_threshold
        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return Node(left=left_subtree, right=right_subtree, feature_index=best_feature, threshold=best_threshold)


    def predict(self, X):
        """
        Make predictions using the trained decision tree.
        """
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        """
        Recursively traverse the decision tree from the root to a leaf node to make predictions.
        """
        if node.predicted_value is not None:
            return node.predicted_value

        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

    def _calculate_info_gain(self, X, y, feature_index, threshold):
        """
        Calculate the information gain by splitting the data based on the feature and threshold.
        """
        parent_entropy = self._calculate_entropy(y)

        left_indices = X[:, feature_index] <= threshold
        right_indices = X[:, feature_index] > threshold

        left_entropy = self._calculate_entropy(y[left_indices])
        right_entropy = self._calculate_entropy(y[right_indices])

        left_weight = len(y[left_indices]) / len(y)
        right_weight = len(y[right_indices]) / len(y)

        info_gain = parent_entropy - (left_weight * left_entropy) - (right_weight * right_entropy)

        return info_gain

    def _calculate_entropy(self, y):
        """
        Calculate the entropy of the target values.
        """
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))  # Add a small value to prevent log(0)
        return entropy

    def _calculate_leaf_value(self, y):
        """
        Calculate the predicted value at a leaf node based on the majority class in the target values.
        """
        unique, counts = np.unique(y, return_counts=True)
        return unique[np.argmax(counts)]


# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an instance of the DecisionTree class
decision_tree = DecisionTree(max_depth=3)

# Fit the decision tree using the training data
decision_tree.fit(X_train, y_train)

# Make predictions on the test data
y_pred = decision_tree.predict(X_test)

# Evaluate the performance of the decision tree using accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
