import pandas as pd


def create_date_features(data: pd.DataFrame):
    data["month"] = data.index.month
    data["day_of_week"] = data.index.dayofweek
    data["is_month_start"] = data.index.is_month_start
    data["is_month_end"] = data.index.is_month_end
    return data


def create_lag_features(data: pd.DataFrame, y: str = "close", lags_min: int = 1, lags_max: int = 20):
    df = data.copy()
    for lag in range(lags_min, lags_max + 1):
        df[f"{y}_{int(lag)}"] = df[y].shift(periods=lag)
    return data
