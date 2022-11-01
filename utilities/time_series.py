import pandas as pd


def create_date_features(data: pd.DataFrame):
    data["month"] = data.index.month
    data["day_of_week"] = data.index.dayofweek
    data["is_month_start"] = data.index.is_month_start
    data["is_month_end"] = data.index.is_month_end
    return data

