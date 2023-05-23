import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
    # Data
    dataset = load_breast_cancer()
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)

    X = df
    y = dataset.target

    # Split Data - test & train data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
    print(X.shape, X_train.shape, X_test.shape)

    # Standarize Data
    print(dataset.data.std())
    scaler = StandardScaler() # z = mean / std
    scaler.fit(X_train)

    X_train_standardized = scaler.transform(X_train)
    X_test_standardized = scaler.transform(X_test)
    print(X_test_standardized)
    print(X_test_standardized.std())