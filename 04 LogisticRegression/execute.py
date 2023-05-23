from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score

from datasets.Standarization import StandardScaler
from datasets.split_data import train_test_validation_split
from utlities.metrics import recall, precision, f1_score
from models.LogisticRegression import LogisticRegression
from sklearn.datasets import load_digits

if __name__ == '__main__':
    # Load the digits dataset
    data = load_digits()
    X, y = data.data, data.target

    # Split the dataset into train and test sets
    X_train, X_test, X_val, y_train, y_test, y_val = train_test_validation_split(X, y, test_size=0.2, val_size=0.2)

    # Create an instance of LogisticRegression
    model = LogisticRegression()

    # Standarize data
    scaler = StandardScaler()
    scaler.fit(X)
    X_std = scaler.transform(X)
    scaler.fit(y)
    y_std = scaler.transform(y)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall(y_test, y_pred)
    precision = precision(y_test, y_pred)
    f1_score = f1_score(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Recall:", recall)
    print("Precision:", precision)
    print("F1 Score:", f1_score)

    # Calculate the probability predictions for each class

    # # Compute the ROC curve and AUC for each class
    # fpr = dict()
    # tpr = dict()
    # roc_auc = dict()
    # for class_label in range(model.K):
    #     fpr[class_label], tpr[class_label], _ = roc_curve(y_test, y_pred_proba[:, class_label], pos_label=class_label)
    #     roc_auc[class_label] = auc(fpr[class_label], tpr[class_label])
    #
    # # Plot the ROC curve for each class
    # plt.figure()
    # for class_label in range(model.K):
    #     plt.plot(fpr[class_label], tpr[class_label], label=f"Class {class_label} (AUC = {roc_auc[class_label]:.2f})")
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic')
    # plt.legend(loc='lower right')
    # plt.show()
