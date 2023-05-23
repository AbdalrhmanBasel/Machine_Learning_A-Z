from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

from PracticeModel.Task_2.DecisionTree import DecisionTree
from datasets.dataset_split_functions import train_test_split

if __name__ == '__main__':
    # import Data
    df = load_iris()

    # Set features & target values
    X = df.data
    y = df.target

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create an instance of the DecisionTree class
    decision_tree = DecisionTree()

    # Fit the decision tree using the training data
    decision_tree.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = decision_tree.predict(X_test)

    # Evaluate the performance of the decision tree using accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)