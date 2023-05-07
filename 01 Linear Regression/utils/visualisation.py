import plotly.graph_objects as go
import numpy as np
import plotly.offline as pyo


class Visualisation():

    @staticmethod
    def visualise_predicted_trace(prediction: np.ndarray, inputs: np.ndarray, targets: np.ndarray, plot_title=''):
        """
        Visualizes predicted trace and targets using Plotly.

        :param prediction: model prediction based on inputs (oy for one trace)
        :param inputs: inputs variables (ox for both)
        :param targets: target variables (oy for one trace)
        :param plot_title: plot title
        """
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=inputs, y=prediction, mode='lines', name='Prediction'))
        fig.add_trace(go.Scatter(x=inputs, y=targets, mode='markers', name='Targets'))
        fig.update_layout(title=plot_title, xaxis_title='Input', yaxis_title='Output')
        pyo.plot(fig, output_type='file')

    @staticmethod
    def visualise_best_models(models, inputs, targets, plot_title=''):
        """
        Visualizes the best models using Plotly.

        :param models: a list of linear regression models
        :param inputs: inputs variables (ox for both)
        :param targets: target variables (oy for one trace)
        :param plot_title: plot title
        """
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=inputs, y=targets, mode='markers', name='Targets'))

        for i, model in enumerate(models):
            prediction = model(inputs)
            fig.add_trace(go.Scatter(x=inputs, y=prediction, mode='lines', name=f'Model {i + 1}'))

        fig.update_layout(title=plot_title, xaxis_title='Input', yaxis_title='Output')
        pyo.plot(fig, output_type='file')
