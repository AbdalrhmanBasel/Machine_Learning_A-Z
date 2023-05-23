import numpy as np

from models.LinearRegression1 import LinearRegression
from models.PolynomialRegression import PolynomialRegression
from datasets.dataset_preparation import DatasetPreparation
from configs.cfg import cfg
from utils.metrics import MSE
from utils.visualisation import Visualisation

if __name__ == '__main__':
    # Dataset
    data_path = './configs/advertising_data.csv'
    dataset = DatasetPreparation(data_path, cfg=cfg)
    data = dataset()

    # Splitting Data
    X_train = data['inputs']['train']
    y_train = data['targets']['train']

    X_test = data['inputs']['test']
    y_test = data['targets']['test']

    X_valid = data['inputs']['valid']
    y_valid = data['targets']['valid']

    # Predicting New Data
    model = LinearRegression(cfg=cfg)
    # model = PolynomialRegression(cfg.py.py=cfg.py.py)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # print(y_pred)

    mse = MSE(y_pred, y_test)
    print(f"Mean Squared Error: {mse}")

    # Visualize The Model
    Graph = Visualisation()
    Graph.visualise_predicted_trace(y_pred, X_test, y_test, plot_title='Predicted Trace and Targets')


