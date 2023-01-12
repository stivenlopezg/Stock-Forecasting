import copy
import numpy as np
import pandas as pd
from itertools import product
from keras.models import Sequential
from keras.backend import clear_session
from keras.callbacks import EarlyStopping
from sklearn.model_selection import TimeSeriesSplit
from keras.layers import SimpleRNN, LSTM, GRU, Dropout, Dense, Bidirectional

tscv = TimeSeriesSplit(n_splits=5)

early_stopping = EarlyStopping(monitor="val_mse", patience=10)


def combinator(items, r: int = 1):
    cmb = [i for i in product(*items, repeat=r)]
    return cmb


def print_best_hyperparameters(hyperparameters: dict):
    for key, value in hyperparameters.items():
        print(f"{key}: {value}")


def create_rnn(hp):
    model = Sequential()
    model.add(layer=SimpleRNN(units=hp.Int(name="input_unit", min_value=32, max_value=128, step=32),
                              activation=hp.Choice(name="activation", values=["relu", "selu", "sigmoid"]),
                              return_sequences=True, input_shape=(1, 34)))
    model.add(layer=Dropout(rate=hp.Float(name="dropout_rate", min_value=0.2, max_value=0.5, step=0.1)))
    for i in range(hp.Int(name="n_layers", min_value=1, max_value=4)):
        model.add(layer=SimpleRNN(units=hp.Int(name=f"rnn_{i}_units", min_value=32, max_value=128, step=32),
                                  activation=hp.Choice(name="activation", values=["relu", "selu", "sigmoid"]),
                                  return_sequences=True))
    model.add(layer=SimpleRNN(units=hp.Int(name=f"rnn_{i + 1}_units", min_value=32, max_value=128, step=32),
                              activation=hp.Choice(name="activation", values=["relu", "selu", "sigmoid"])))
    model.add(layer=Dropout(rate=hp.Float(name="dropout_rate", min_value=0.2, max_value=0.5, step=0.1)))
    model.add(layer=Dense(units=1, activation="linear"))
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mse"])
    return model


def create_lstm(hp):
    model = Sequential()
    model.add(layer=LSTM(units=hp.Int(name="input_unit", min_value=32, max_value=128, step=32),
                         activation=hp.Choice(name="activation", values=["relu", "selu", "sigmoid"]),
                         return_sequences=True, input_shape=(1, 34)))
    model.add(layer=Dropout(rate=hp.Float(name="dropout_rate", min_value=0.2, max_value=0.5, step=0.1)))
    for i in range(hp.Int(name="n_layers", min_value=1, max_value=4)):
        model.add(layer=LSTM(units=hp.Int(name=f"lstm_{i}_units", min_value=32, max_value=128, step=32),
                             activation=hp.Choice(name="activation", values=["relu", "selu", "sigmoid"]),
                             return_sequences=True))
    model.add(layer=LSTM(units=hp.Int(name=f"lstm_{i}_units", min_value=32, max_value=128, step=32),
                         activation=hp.Choice(name="activation", values=["relu", "selu", "sigmoid"])))
    model.add(layer=Dropout(rate=hp.Float(name="dropout_rate", min_value=0.2, max_value=0.5, step=0.1)))
    model.add(layer=Dense(units=1, activation="linear"))
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mse"])
    return model


def create_gru(hp):
    model = Sequential()
    model.add(layer=GRU(units=hp.Int(name="input_unit", min_value=32, max_value=128, step=32),
                        activation=hp.Choice(name="activation", values=["relu", "selu", "sigmoid"]),
                        return_sequences=True, input_shape=(1, 34)))
    model.add(layer=Dropout(rate=hp.Float(name="dropout_rate", min_value=0.2, max_value=0.5, step=0.1)))
    for i in range(hp.Int(name="n_layers", min_value=1, max_value=4)):
        model.add(layer=GRU(units=hp.Int(name=f"gru_{i}_units", min_value=32, max_value=128, step=32),
                            activation=hp.Choice(name="activation", values=["relu", "selu", "sigmoid"]),
                            return_sequences=True))
    model.add(layer=GRU(units=hp.Int(name=f"gru_{i}_units", min_value=32, max_value=128, step=32),
                        activation=hp.Choice(name="activation", values=["relu", "selu", "sigmoid"])))
    model.add(layer=Dropout(rate=hp.Float(name="dropout_rate", min_value=0.2, max_value=0.5, step=0.1)))
    model.add(layer=Dense(units=1, activation="linear"))
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mse"])
    return model


def create_bidirectionallstm(hp):
    model = Sequential()
    model.add(layer=Bidirectional(LSTM(units=hp.Int(name="input_unit", min_value=32, max_value=128, step=32),
                                       activation=hp.Choice(name="activation", values=["relu", "selu", "sigmoid"]),
                                       return_sequences=True, input_shape=(1, 34))))
    model.add(layer=Dropout(rate=hp.Float(name="dropout_rate", min_value=0.2, max_value=0.5, step=0.1)))
    for i in range(hp.Int(name="n_layers", min_value=1, max_value=4)):
        model.add(layer=Bidirectional(LSTM(units=hp.Int(name=f"bilstm_{i}_units", min_value=32, max_value=128, step=32),
                                           activation=hp.Choice(name="activation", values=["relu", "selu", "sigmoid"]),
                                           return_sequences=True)))
    model.add(layer=Bidirectional(LSTM(units=hp.Int(name=f"bilstm_{i}_units", min_value=32, max_value=128, step=32),
                                       activation=hp.Choice(name="activation", values=["relu", "selu", "sigmoid"]))))
    model.add(layer=Dropout(rate=hp.Float(name="dropout_rate", min_value=0.2, max_value=0.5, step=0.1)))
    model.add(layer=Dense(units=1, activation="linear"))
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mse"])
    return model


def create_stacked_fts(estimator, data: pd.DataFrame, close: pd.Series):
    predictions = pd.Series(index=data.index)
    models_list = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(data, close)):
        fold_model = copy.deepcopy(estimator)
        if callable(getattr(fold_model, "search", None)):
            X_train = data.loc[train_idx, :].values.reshape(-1, 1, data.shape[1])
            y_train = close.loc[train_idx]
            X_test = data.loc[test_idx, :].values.reshape(-1, 1, data.shape[1])
            clear_session()
            fold_model.search(X_train, y_train, epochs=1000, validation_split=0.3, callbacks=[early_stopping])
            model = fold_model.get_best_models(num_models=1)[0]
            predictions.loc[test_idx] = model.predict(X_test, verbose=0).ravel()
            models_list.append(model)
        else:
            fold_model.fit(data.loc[train_idx, :], close.loc[train_idx])
            predictions.loc[test_idx] = fold_model.predict(data.loc[test_idx, :]).flatten()
            models_list.append(fold_model)
    return predictions, models_list


def train_stacking(data: pd.DataFrame, close: pd.Series, base_models: list):
    trained_models = {}

    df = data.copy()

    for idx, model in enumerate(base_models):
        if model.__class__.__name__ == "Ridge":
            predictions, fold_models = create_stacked_fts(estimator=model, data=data.iloc[:, 2:], close=close)
        else:
            predictions, fold_models = create_stacked_fts(estimator=model, data=data, close=close)
        df.loc[predictions.index.tolist(), f"{model.__class__.__name__}_forecast"] = predictions
        trained_models[f"{model.__class__.__name__}_forecast"] = fold_models
    return df, trained_models


def transform_test(data: pd.DataFrame, trained_models):
    df = data.copy()
    for ft, model_list in trained_models.items():
        if ft.startswith("Ridge"):
            preds = [model.predict(data.iloc[:, 2:]) for model in model_list]
            avg_preds = np.array([sum(x) for x in zip(*preds)]) / len(model_list)
        else:
            preds = [model.predict(data) for model in model_list]
            avg_preds = np.array([sum(x) for x in zip(*preds)]) / len(model_list)
        df[ft] = avg_preds
    return df
