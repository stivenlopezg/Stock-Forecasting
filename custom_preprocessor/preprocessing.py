import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class AutorregresiveFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, target: str = "close", lags_min: int = 1, lags_max: int = 30):
        self.target = target
        self.lags_min = lags_min
        self.lags_max = lags_max

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame):
        for lag in range(self.lags_min, self.lags_max + 1):
            X[f"{self.target}_{int(lag)}"] = X[self.target].shift(periods=lag)
        X.dropna(inplace=True)
        return X


class SMAFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, target: str = "close", periods=None):
        if periods is None:
            periods = []
        self.target = target
        self.periods = periods

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame):
        for period in self.periods:
            X[f"sma_{int(period)}"] = X[self.target].rolling(window=period).mean()
        return X


class GetDummies(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list):
        if not isinstance(columns, list):
            self.columns = [columns]
        else:
            self.columns = columns

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.get_dummies(X, columns=self.columns)
        return X
