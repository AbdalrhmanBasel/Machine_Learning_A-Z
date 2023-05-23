import math


class DecisionTreeRegressor:

    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        self._features = X
        self._target = y

        self._tree = self._build_tree()

    def predict(self, X):
        predictions = []
        for x in X:
            predictions.append(self._predict(x))

        return predictions

    def _build_tree(self):
        root = DecisionTreeNode(self._features, self._target)

        self._build_tree_recursively(root)

        return root

    def _build_tree_recursively(self, node):
        if len(node.features) <= self.min_samples_leaf:
            node.prediction = node.target.mean()
            return

        best_feature, best_split_value = self._find_best_split(node)

        node.left = DecisionTreeNode(node.features[node.features[best_feature] <= best_split_value], node.target[node.features[best_feature] <= best_split_value])
        node.right = DecisionTreeNode(node.features[node.features[best_feature] > best_split_value], node.target[node.features[best_feature] > best_split_value])

        self._build_tree_recursively(node.left)
        self._build_tree_recursively(node.right)

    def _find_best_split(self, node):
        best_gain = -float("inf")
        best_feature = None
        best_split_value = None

        for feature in node.features:
            for split_value in np.unique(node.target):
                gain = self._information_gain(node.target, feature, split_value)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_split_value = split_value

        return best_feature, best_split_value

    def _information_gain(self, target, feature, split_value):
        target_1d = target.ravel()

        entropy_before = self._entropy(target_1d)

        target_left = target_1d[target_1d[feature] <= split_value]
        target_right = target_1d[target_1d[feature] > split_value]

        entropy_after_left = self._entropy(target_left)
        entropy_after_right = self._entropy(target_right)

        gain = entropy_before - (len(target_left) / len(target) * entropy_after_left + len(target_right) / len(
            target) * entropy_after_right)

        return gain

    def _entropy(self, target):
        counts = np.unique(target, return_counts=True)[1]

        entropy = 0
        for count in counts:
            p = count / len(target)

            if p > 0:
                entropy += -p * math.log2(p)

        return entropy


class DecisionTreeNode:

    def __init__(self, features, target):
        self.features = features
        self.target = target
        self.prediction = None
        self.left = None
        self.right = None

import numpy as np

# Generate some data
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array([3, 4, 5, 6])

# Create a DecisionTreeRegressor
regressor = DecisionTreeRegressor()

# Fit the regressor to the data
regressor.fit(X, y)

# Predict some values
# Predict some values
predictions = regressor.predict(np.array([[9, 10], [11, 12]]))

# Print the predictions
print(predictions)

