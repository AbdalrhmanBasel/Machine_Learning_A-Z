

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from models.DecisionTreeClassifier import DecisionTreeClassifier

if __name__ == '__main__':

    # Load the digits dataset
    digits = load_digits()
    X = digits.data
    y = digits.target

    # Split the dataset into train, test, and validation sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    # Create and train the decision tree classifier
    tree = DecisionTreeClassifier(max_depth=5, entropy_threshold=0.1, element_threshold=5)
    tree.fit(X_train, y_train)

    # Predict the labels for validation and test sets
    y_valid_pred = tree.predict(X_valid)
    y_test_pred = tree.predict(X_test)

    # Calculate accuracy and build confusion matrix for validation set
    valid_accuracy = accuracy_score(y_valid, y_valid_pred)
    valid_cm = confusion_matrix(y_valid, y_valid_pred)

    # Calculate accuracy and build confusion matrix for test set
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_cm = confusion_matrix(y_test, y_test_pred)

    print("Validation Accuracy:", valid_accuracy)
    print("Validation Confusion Matrix:\n", valid_cm)

    print("Test Accuracy:", test_accuracy)
    print("Test Confusion Matrix:\n", test_cm)