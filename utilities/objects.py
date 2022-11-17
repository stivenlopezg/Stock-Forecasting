import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


class MatrixObject:
    def __init__(self, X: pd.DataFrame, y: pd.Series):
        self.X = X
        self.y = y


class Model(BaseEstimator, RegressorMixin):
    def __init__(self, estimator):
        self.X = None
        self.y = None
        self.estimator = estimator

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        self.X = X
        self.y = y
        return self.estimator.fit(X=self.X, y=self.y)

    def predict(self, X: pd.DataFrame):
        return self.estimator.predict(X)
