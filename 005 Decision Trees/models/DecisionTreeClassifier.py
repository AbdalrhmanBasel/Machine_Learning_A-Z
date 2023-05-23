import numpy as np


class Node:
    def __init__(self, feature_idx=None, threshold=None, value=None, left=None, right=None, leaf=False, label=None):
        self.feature_idx = feature_idx  # Index of the feature to split on
        self.threshold = threshold  # Threshold value for the split
        self.value = value  # Majority class value at this node
        self.left = left  # Left child node
        self.right = right  # Right child node
        self.leaf = leaf  # True if this node is a leaf node
        self.label = label  # Class label for leaf nodes


class DecisionTreeClassifier:
    def __init__(self, max_depth=None, entropy_threshold=None, element_threshold=None):
        self.max_depth = max_depth  # Maximum depth of the tree
        self.entropy_threshold = entropy_threshold  # Entropy threshold to create a terminal node
        self.element_threshold = element_threshold  # Number of elements threshold to create a terminal node
        self.root = None  # Root node of the decision tree

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        # Check termination conditions
        if (self.max_depth is not None and depth >= self.max_depth) or \
                (self._calculate_entropy(y) <= self.entropy_threshold) or \
                (self.element_threshold is not None and len(y) <= self.element_threshold):
            return Node(leaf=True, label=self._majority_class(y))

        n_features = X.shape[1]
        best_feature, best_threshold = None, None
        best_info_gain = -np.inf

        # Find the best feature and threshold for splitting
        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                left_indices = X[:, feature_idx] <= threshold
                right_indices = X[:, feature_idx] > threshold
                info_gain = self._information_gain(y, y[left_indices], y[right_indices])
                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_feature = feature_idx
                    best_threshold = threshold

        if best_feature is None:
            return Node(leaf=True, label=self._majority_class(y))

        left_indices = X[:, best_feature] <= best_threshold
        right_indices = X[:, best_feature] > best_threshold

        left_node = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_node = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return Node(feature_idx=best_feature, threshold=best_threshold, left=left_node, right=right_node)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.leaf:
            return node.label

        if x[node.feature_idx] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

    @staticmethod
    def _calculate_entropy(y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities))

    @staticmethod
    def _information_gain(parent, left_child, right_child):
        parent_entropy = DecisionTreeClassifier._calculate_entropy(parent)
        left_entropy = DecisionTreeClassifier._calculate_entropy(left_child)
        right_entropy = DecisionTreeClassifier._calculate_entropy(right_child)
        left_weight = len(left_child) / len(parent)
        right_weight = len(right_child) / len(parent)
        return parent_entropy - (left_weight * left_entropy + right_weight * right_entropy)

    @staticmethod
    def _majority_class(y):
        unique, counts = np.unique(y, return_counts=True)
        return unique[np.argmax(counts)]



