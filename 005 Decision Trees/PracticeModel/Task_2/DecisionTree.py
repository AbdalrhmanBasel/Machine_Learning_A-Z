import numpy as np


class Node:
    """
    Define the Node class:
        1. Create a class named Node to represent a node in the decision tree.
        2. Define attributes for feature index, threshold value, predicted value at the leaf node, and left/right subtrees.
    """

    def __init__(self, left=None, right=None, feature_index=None, threshold=None, predicted_value=None):
        self.left = left  # Left subtree
        self.right = right  # Right subtree
        self.feature_index = feature_index  # Index of the feature to split on
        self.threshold = threshold  # Threshold value for splitting
        self.predicted_value = predicted_value  # Predicted value at the leaf node


class DecisionTree:
    def __init__(self, max_depth=None, min_simple_split=2):
        self.max_depth = max_depth
        self.min_simple_split = min_simple_split
        self.root = None

    def fit(self, X, y):
        """
        Implement the fit method:
            Define the fit method in the DecisionTree class to train the decision tree.
            Accept training data X and y as input.
            Inside the fit method, call the private _build_tree method to construct the decision tree recursively.
        """

        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        """
        Implement the _build_tree method:
            1. Define the private method _build_tree in the DecisionTree class.
            2. Accept training data X, y, and depth as input.
            3. Implement the stopping conditions for recursion:
                - Check if the depth has reached the max_depth.
                - Check if the number of samples is less than min_samples_split.
                - Check if all target values are the same.
            7. If any stopping condition is met, create a leaf node and assign the predicted value.
                Otherwise, find the best feature and threshold to split on based on information gain.
            8. Split the data based on the best feature and threshold into left and right subsets.
            9. Recursively call _build_tree to construct the left and right subtrees.
            10. Return the current node with the assigned feature index, threshold, left, and right subtrees.
        """
        # Stopping Conditions for the tree
        if depth >= self.max_depth or len(X) < self.min_simple_split or len(np.unique(y)) == 1:
            return Node(value=self._calculate_leaf_value(y))  # Create a leaf node and assign the predicted value.

        #
        n_features = X.shape[1]  # calculates the number of features in the input dataset X
        best_feature, best_threshold = None, None
        best_info_gain = -np.inf  # initialize the variable with negative infinity (-np.inf)

        # Split the data based on the best feature and threshold into left and right subsets.

        for feature_index in range(n_features):
            unique_values = np.unique[X[:, feature_index]]
            for threshold in unique_values:
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
        Implement the predict method:
            1. Define the predict method in the DecisionTree class to make predictions using the trained decision tree.
            2. Accept test data X as input.
            3. Iterate over each instance in X and call the private _traverse_tree method to traverse the decision tree and get the predicted value.
            4. Return the predicted values as a numpy array.
        """
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        """
        This function to walk recursively from the root to the leaf to predict the leaf value.

        Implement the _traverse_tree method:
            1. Define the private method _traverse_tree in the DecisionTree class.
            2. Accept an instance x and a node as input.
            3. Check if the current node is a leaf node (i.e., it has a predicted value).
                - If it is a leaf node, return the predicted value.
                - Otherwise, compare the feature value of x with the threshold of the current node.
            5. If the feature value is less than or equal to the threshold, recursively call _traverse_tree with the left subtree as the current node.
            6. If the feature value is greater than the threshold, recursively call _traverse_tree with the right subtree as the current node.
        """
        if node.value is not None:
            return node.value

        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

    def _calculate_information_gain(self, X, y, feature_index, threshold):
        """
        Implement the _calculate_info_gain method:
            1. Define the private method _calculate_info_gain in the DecisionTree class.
            2. Accept training data X, target values y, feature index, and threshold as input.
            3. Calculate the entropy of the parent node using the _calculate_entropy method.
            4. Split the data based on the feature index and threshold.
            5. Calculate the entropy of the left and right child nodes using the _calculate_entropy method.
            6. Calculate the information gain by subtracting the weighted average of child node entropies from the parent entropy.
            7. Return the information gain.

        Example:
            1. Calculate the entropy of the target variable (PlayTennis):
            Total instances: 14
            Positive instances (PlayTennis = Yes): 9
            Negative instances (PlayTennis = No): 5
            Entropy = -((9/14) * log2(9/14) + (5/14) * log2(5/14)) = 0.940
            2. Calculate the information gain for each feature:
            a) Outlook:

            Sunny instances: 5 (2 Yes, 3 No)
            Overcast instances: 4 (4 Yes, 0 No)
            Rainy instances: 5 (3 Yes, 2 No)
            Outlook entropy = -((2/5) * log2(2/5) + (3/5) * log2(3/5)) = 0.971
            Information gain = 0.940 - ((5/14) * 0.971 + (4/14) * 0 + (5/14) * 0.971) = 0.246
        """
        parent_entropy = self._calculate_entropy(y)

        left_indices = X[:, feature_index] <= threshold
        right_indices = X[:, feature_index] > threshold

        # Calculate the entropy of the left and right child nodes using the _calculate_entropy method.
        left_entropy = self._calculate_entropy(y[left_indices])
        right_entropy = self._calculate_entropy(y[left_indices])

        # Calculate the information gain by subtracting the weighted average
        # of child node entropies from the parent entropy.

        left_weight = len(y[left_indices]) / len(y)
        right_weight = len(y(right_indices)) / len(y)

        info_gain = parent_entropy - (left_weight * left_entropy) - (right_weight * right_entropy)

        return info_gain

    def _calculate_entropy(self, y):
        """
        Define the private method _calculate_entropy in the DecisionTree class.
        Accept target values y as input.
        Calculate the count and probability of each unique target value.
        Calculate the entropy using the entropy formula (-sum(p * log2(p))).
        Return the entropy value.
        """
        _, counts = np.unique(y, return_counts=True)
        probabilities  = counts / len(y)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))  # 1e - 10 to avoid taking log of zero &
        # prevent NAN results
        return entropy

    def _calculate_leaf_value(self,y):
        """
        Define the private method _calculate_leaf_value in the DecisionTree class.
        Accept target values y as input.
        Calculate the count of each unique target value.
        Return the value with the highest count as the predicted value at the leaf node.
        """
        unique, counts = np.unique(y, return_counts=True)
        return unique[np.argmax(counts)]
