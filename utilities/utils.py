from itertools import product
from keras.models import Sequential
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
                              return_sequences=True, input_shape=(1, 33)))
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
                         return_sequences=True, input_shape=(1, 33)))
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
                        return_sequences=True, input_shape=(1, 33)))
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
                                       return_sequences=True, input_shape=(1, 33))))
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