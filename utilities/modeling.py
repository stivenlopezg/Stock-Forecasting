import copy
import numpy as np
import pandas as pd
from keras.backend import clear_session
from keras.callbacks import EarlyStopping
from sklearn.model_selection import TimeSeriesSplit

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from custom_preprocessor.preprocessing import date_features, create_date_features, create_autorregresive_features,\
                                              create_sma_features

tscv = TimeSeriesSplit(n_splits=5)

early_stopping = EarlyStopping(monitor="val_mse", patience=10)

def create_features(data: pd.DataFrame,
                    target: str, lags: int = 20,
                    periods: list = None, **kwargs) -> tuple[pd.DataFrame, pd.Series]:
    df = create_date_features(data=data)
    for d_ft in date_features:
        if d_ft == "month":
            df[d_ft] = pd.Categorical(df[d_ft], categories=[int(i) for i in range(1, 13)])
        else:
            categories = df[d_ft].unique().tolist()
            df[d_ft] = pd.Categorical(df[d_ft], categories=categories)
    df = create_autorregresive_features(data=df, lags_max=lags)
    df = create_sma_features(data=df, periods=periods, **kwargs)
    df = pd.get_dummies(data=df, columns=date_features, drop_first=False)
    df.dropna(inplace=True)
    close = df.pop(target)
    return df, close


def get_best_hyperparameters(estimator,
                             data: pd.DataFrame,
                             target: str,
                             parameters: dict, ts_cv,
                             tuner: str,
                             lags_max_candidates: list,
                             periods_candidates: list = None, **kwargs):
    if tuner not in ["grid_search", "random_search"]:
        raise ValueError("El parámetro tuner debe ser grid_search o random_search, no se soporta otro método.")
    scores = []
    hyperparameters = []
    for lag in lags_max_candidates:
        if periods_candidates is not None:
            for periods in periods_candidates:
                df, close = create_features(data=data, target=target, lags=lag, periods=periods)
                if tuner == "grid_search":
                    model = GridSearchCV(estimator=estimator,
                                         param_grid=parameters,
                                         scoring="neg_mean_absolute_percentage_error",
                                         cv=ts_cv, n_jobs=-1, **kwargs).fit(X=df, y=close)
                else:
                    model = RandomizedSearchCV(estimator=estimator,
                                               param_distributions=parameters,
                                               n_iter=10, scoring="neg_mean_absolute_percentage_error",
                                               cv=ts_cv, n_jobs=-1, **kwargs).fit(X=df, y=close)
                scores.append(np.round(-1 * model.best_score_, decimals=6))
                best_params = model.best_params_
                for key, value in zip(["lag_max", "periods"], [lag, periods]):
                    best_params[key] = value
                hyperparameters.append(best_params)
    best_hyperparameters = hyperparameters[scores.index(min(scores))]
    return best_hyperparameters, min(scores)


def fit_best_model(estimator, data: pd.DataFrame, target: str, parameters: dict, **fit_params):
    if "lag_max" not in parameters.keys():
        raise ValueError(f"ValueError: parameters dict debe tener los keys: lag_max y periods.")
    elif "periods" not in parameters.keys():
        raise ValueError(f"ValueError: parameters dict debe tener los keys: lag_max y periods.")
    df, close = create_features(data=data, target=target,
                                lags=parameters.get("lag_max"), periods=parameters.get("periods"))
    model = estimator.fit(df, close, **fit_params)
    return model


def create_train_fts_for_stacking(estimator, data: pd.DataFrame, close: pd.Series):
    predictions = pd.Series(index=data.index)
    fitted_models = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(data, close)):
        fold_model = copy.deepcopy(estimator)
        if callable(getattr(fold_model, "search", None)):
            clear_session()
            n_fts = data.shape[1]
            fold_model.search(data.loc[train_idx, :].values.reshape(-1, 1, n_fts),
                              close.loc[train_idx],
                              epochs=1000, validation_split=0.3, callbacks=[early_stopping])
            model = fold_model.get_best_models(num_models=1)[0]
            predictions.loc[test_idx] = model.predict(data.loc[test_idx, :].values.reshape(-1, 1, n_fts)).ravel()
            fitted_models.append(model)
        else:
            fold_model.fit(data.loc[train_idx, :], close.loc[train_idx])
            predictions.loc[test_idx] = fold_model.predict(data.loc[test_idx, :]).flatten()
            fitted_models.append(fold_model)
    return predictions, fitted_models


def create_test_fts_for_stacking(data: pd.DataFrame, trained_models: dict):
    df = data.copy()
    for key, models in trained_models.items():
        if models[0].__class__.__name__ == "Sequential":
            preds = [model.predict(df.values.reshape(-1, 1, df.shape[1])) for model in models]
        else:
            preds = [model.predict(data) for model in models]
        df[key] = np.array([sum(x) for x in zip(*preds)]) / len(models)
    return df


def fit_stacking_model(data: pd.DataFrame, close: pd.Series, base_models: list):
    fitted_models = {}

    df = data.copy()

    for idx, model in enumerate(base_models):
        preds, fold_models = create_train_fts_for_stacking(estimator=model, data=data, close=close)
        if fold_models[0].__class__.__name__ == "Sequential":
            df.loc[preds.index, f"{fold_models[0].get_layer(index=0).name}_forecast"] = preds
            fitted_models[f"{fold_models[0].get_layer(index=0).name}_forecast"] = fold_models
        else:
            df.loc[preds.index.tolist(), f"{model.__class__.__name__}_forecast"] = preds
            fitted_models[f"{model.__class__.__name__}_forecast"] = fold_models
    return df, fitted_models


