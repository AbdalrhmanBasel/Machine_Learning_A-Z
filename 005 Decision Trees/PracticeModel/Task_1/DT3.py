import numpy as np
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split
from utilities.Evaluation_Metrics.classification_metrics import accuracy,recall,precision,f1_score


def entropy(y):
    hist = np.bincount(y)
    ps = hist / len(y)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature  # Splitting feature at this node
        self.threshold = threshold  # Threshold value for the splitting feature
        self.left = left  # Left child node
        self.right = right  # Right child node
        self.value = value  # Predicted value at the leaf node

    def is_leaf_node(self):
        """
        Check if the node is a leaf node.
        """
        return self.value is not None


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_feats=None):
        self.min_samples_split = min_samples_split  # Minimum number of samples required to split a node
        self.max_depth = max_depth  # Maximum depth of the tree
        self.n_feats = n_feats  # Number of features to consider for splitting
        self.root = None  # Root node of the decision tree

    def fit(self, X, y):
        """
        Build the decision tree using the training data.

        Args:
            X (numpy array): Feature matrix of shape (n_samples, n_features).
            y (numpy array): Target values of shape (n_samples,).
        """
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        self.root = self._grow_tree(X, y)

    def predict(self, X):
        """
        Make predictions for the input samples.

        Args:
            X (numpy array): Feature matrix of shape (n_samples, n_features).

        Returns:
            numpy array: Predicted target values.
        """
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _grow_tree(self, X, y, depth=0):
        """
        Recursively build the decision tree.

        Args:
            X (numpy array): Feature matrix of shape (n_samples, n_features).
            y (numpy array): Target values of shape (n_samples,).
            depth (int): Current depth of the tree.

        Returns:
            Node: Root node of the constructed decision tree.
        """
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Stopping conditions
        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            # Create a leaf node and assign the most frequent class label as the predicted value
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)

        # Greedily select the best split according to information gain
        best_feat, best_thresh = self._best_criteria(X, y, feat_idxs)

        # Grow the children that result from the split
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)

        # Create an internal node with the best feature and threshold
        return Node(feature=best_feat, threshold=best_thresh, left=left, right=right)

    def _best_criteria(self, X, y, feat_idxs):
        """
        Find the best feature and threshold for splitting the data based on information gain.

        Args:
            X (numpy array): Feature matrix of shape (n_samples, n_features).
            y (numpy array): Target values of shape (n_samples,).
            feat_idxs (numpy array): Indices of features to consider.

        Returns:
            int: Index of the best feature.
            float: Best threshold value.
        """
        best_gain = -1
        split_idx, split_thresh = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold

        return split_idx, split_thresh

    def _information_gain(self, y, X_column, split_thresh):
        """
        Calculate the information gain after a split.

        Args:
            y (numpy array): Target values of shape (n_samples,).
            X_column (numpy array): Feature column values corresponding to the splitting feature.
            split_thresh (float): Threshold value for the splitting feature.

        Returns:
            float: Information gain.
        """
        # Compute the entropy of the parent node
        parent_entropy = entropy(y)

        # Generate the split
        left_idxs, right_idxs = self._split(X_column, split_thresh)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # Compute the entropy of the left and right child nodes
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = entropy(y[left_idxs]), entropy(y[right_idxs])

        # Compute the weighted average of the entropy for the children
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        # Information gain is the difference in entropy before and after the split
        ig = parent_entropy - child_entropy
        return ig

    def _split(self, X_column, split_thresh):
        """
        Split the data based on the threshold value.

        Args:
            X_column (numpy array): Feature column values corresponding to the splitting feature.
            split_thresh (float): Threshold value for the splitting feature.

        Returns:
            numpy array: Indices of samples on the left side of the split.
            numpy array: Indices of samples on the right side of the split.
        """
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _traverse_tree(self, x, node):
        """
        Traverse the decision tree to make predictions for a single sample.

        Args:
            x (numpy array): Feature vector of shape (n_features,).
            node (Node): Current node being traversed.

        Returns:
            int: Predicted target value.
        """
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

    def _most_common_label(self, y):
        """
        Find the most common class label.

        Args:
            y (numpy array): Target values of shape (n_samples,).

        Returns:
            int: Most common class label.
        """
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common


if __name__ == "__main__":
    data = datasets.load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    clf = DecisionTree(max_depth=10)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy(y_test, y_pred)
    precision = precision(y_test, y_pred)
    recall = recall(y_test, y_pred)
    f1_score = f1_score(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1_score:", f1_score)
