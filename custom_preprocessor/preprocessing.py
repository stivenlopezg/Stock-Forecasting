import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class AutorregresiveFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, lags_min: int = 1, lags_max: int = 30):
        self.lags_min = lags_min
        self.lags_max = lags_max
        self.y = None

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame):
        data = X.copy()
        for lag in range(self.lags_min, self.lags_max + 1):
            data[f"close_{int(lag)}"] = data["close"].shift(periods=lag)
        return data


class SMAFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, periods=None):
        if periods is None:
            periods = []
        self.periods = periods
        self.y = None

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame):
        data = X.copy()
        for period in self.periods:
            data[f"sma_{int(period)}"] = data["close"].rolling(window=period).mean()
        return data


class GetDummies(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list):
        if not isinstance(columns, list):
            self.columns = [columns]
        else:
            self.columns = columns

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        data = X.copy()
        data = pd.get_dummies(data, columns=self.columns)
        return data


class DropNaN(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame):
        data = X.copy()
        data = data.dropna()
        return data


class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame):
        data = X.copy()
        data.drop(columns=self.columns, inplace=True)
        return data
