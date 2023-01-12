import pandas as pd
from utilities.time_series import create_date_features
from sklearn.base import BaseEstimator, TransformerMixin

date_features = ["month", "day_of_week", "is_month_start", "is_month_end"]


class AutorregresiveFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, lags_min: int = 1, lags_max: int = 30):
        self.lags_min = lags_min
        self.lags_max = lags_max

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


class DateFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, categories: dict = None):
        self.categories = categories

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        for ft in [ft for ft in X.columns.tolist() if ft != "close"]:
            self.categories[ft] = X[ft].unique().tolist()
        return self

    def transform(self, X: pd.DataFrame):
        data = X.copy()
        data = create_date_features(data=data)
        return data


class ConvertToCategorical(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_fts: list = None):
        self.categorical_fts = categorical_fts
        self.categories = {}

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        for ft in self.categorical_fts:
            self.categories[ft] = X[ft].unique().tolist()
        return self

    def transform(self, X: pd.DataFrame):
        data = X.copy()
        for ft in self.categorical_fts:
            data[ft] = pd.Categorical(values=data[ft], categories=self.categories[ft])
        return data

