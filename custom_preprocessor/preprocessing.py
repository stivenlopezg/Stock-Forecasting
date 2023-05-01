import pandas as pd

date_features = ["month", "day_of_week", "is_month_start", "is_month_end"]

def create_date_features(data: pd.DataFrame, date_is_index=True):
    df = data.copy()
    if date_is_index:
        df["month"] = df.index.month
        df["day_of_week"] = df.index.dayofweek
        df["is_month_start"] = df.index.is_month_start
        df["is_month_end"] = df.index.is_month_end
    else:
        df["month"] = df["date"].dt.month
        df["day_of_week"] = df["date"].dt.dayofweek
        df["is_month_start"] = df["date"].dt.is_month_start
        df["is_month_end"] = df["date"].dt.is_month_end
    return df

def create_autorregresive_features(data: pd.DataFrame, lags_min: int = 1, lags_max: int = 20, **kwargs):
    df = data.copy()
    for lag in range(lags_min, lags_max + 1):
        df[f"close_{int(lag)}"] = df["close"].shift(periods=lag, **kwargs)
    return df


def create_sma_features(data: pd.DataFrame, periods: list = None, **kwargs):
    if periods is None:
        periods = []
    df = data.copy()
    if len(periods) != 0:
        for period in periods:
            df[f"sma_{int(period)}"] = df["close"].rolling(window=period, **kwargs).mean()
    return df