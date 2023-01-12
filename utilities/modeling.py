import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from custom_preprocessor.preprocessing import date_features
from custom_preprocessor.preprocessing import DateFeatures, AutorregresiveFeatures, SMAFeatures, DropNaN, DropColumns, \
    GetDummies, ConvertToCategorical


def define_preprocessing(lags: int = 20, periods: list = None, date_fts: list = date_features):
    if periods is None:
        periods = [5, 10, 15]
    _preprocessing = Pipeline(steps=[("date_fts", DateFeatures()),
                                     ("to_categorical", ConvertToCategorical(categorical_fts=date_fts)),
                                     ("ar", AutorregresiveFeatures(lags_max=lags)),
                                     ("sma", SMAFeatures(periods=periods)),
                                     ("dropnan", DropNaN()),
                                     ("column_selector", DropColumns(columns=["close"])),
                                     ("dummies", GetDummies(columns=date_fts))])
    return _preprocessing


def transform_data(data: pd.DataFrame, close, lag: int, periods: list):
    _preprocessing = define_preprocessing(lags=lag, periods=periods)
    _train_data = _preprocessing.fit_transform(data, close)
    _train_close = close.loc[_train_data.index.tolist()]
    return _preprocessing, _train_data, _train_close


def get_best_hyperparameters(estimator,
                             data: pd.DataFrame,
                             close: pd.Series,
                             parameters: dict, cv,
                             lags_max_candidates=None,
                             periods_candidates=None):
    if lags_max_candidates is None:
        lags_max_candidates = [20]
    if periods_candidates is None:
        periods_candidates = [[5, 10, 15]]
    scores = []
    hyperparameters = []

    for lag in lags_max_candidates:
        for period in periods_candidates:
            _preprocessing, _train, _train_target = transform_data(data=data,
                                                                   close=close,
                                                                   lag=lag,
                                                                   periods=period)
            if estimator.__class__.__name__ == "Ridge":
                _model = GridSearchCV(estimator=estimator,
                                      param_grid=parameters,
                                      scoring="neg_mean_absolute_percentage_error", cv=cv, n_jobs=-1).fit(
                    _train.iloc[:, 2:],
                    _train_target)
            else:
                _model = GridSearchCV(estimator=estimator,
                                      param_grid=parameters,
                                      scoring="neg_mean_absolute_percentage_error", cv=cv, n_jobs=-1).fit(_train,
                                                                                                          _train_target)
            scores.append(np.round(-1 * _model.best_score_, decimals=6))
            best_params = _model.best_params_
            for key, value in zip(["lag_max", "periods"], [lag, period]):
                best_params[key] = value
            hyperparameters.append(best_params)
    best_hyperparameters = hyperparameters[scores.index(min(scores))]
    return best_hyperparameters, min(scores)


def fit_model(estimator, data: pd.DataFrame, close, lag: int, period: list):
    _preprocessing, _train_data, _train_close = transform_data(data=data,
                                                               close=close,
                                                               lag=lag,
                                                               periods=period)
    if estimator.__class__.__name__ == "Ridge":
        _model = estimator.fit(_train_data.iloc[:, 2:], _train_close)
    else:
        _model = estimator.fit(_train_data, _train_close)
    return _preprocessing, _model
