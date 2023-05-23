from sklearn import datasets
from Datasets import split_data as sd
from models.BinaryClassification import LogisticRegression

if __name__ == "__main__":
    # import some data to play with
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # we only take the first two features.
    y = iris.target

    X_train, X_test, y_train, y_test = sd.train_test_split(X, y, test_size=0.3, random_state=42)

    # Model
    model = LogisticRegression(learning_rate=0.1, n_iterations=10000, fit_intercept=True, verbose=True)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test, 0.5)
    print(y_pred)
