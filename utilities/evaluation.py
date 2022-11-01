import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error


class RegressionEvaluator(object):
    def __init__(self, predicted: pd.Series, observed: pd.Series):
        self.predicted = predicted
        self.observed = observed
        self.metrics = None

    def calculate_metrics(self):
        self.metrics = {"rmse": mean_squared_error(y_true=self.observed, y_pred=self.predicted, squared=False),
                        "mae": mean_absolute_error(y_true=self.observed, y_pred=self.predicted),
                        "mape": mean_absolute_percentage_error(y_true=self.observed, y_pred=self.predicted)}
        return self.metrics

    def print_metrics(self):
        if self.metrics is None:
            self.calculate_metrics()
        print(f"El RMSE es: {self.metrics['rmse']}")
        print(f"El MAE es: {self.metrics['mae']}")
        print(f"El MAPE es: {self.metrics['mape']}")
