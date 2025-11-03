import pandas as pd


def get_min_data(df: pd.DataFrame) -> pd.Timestamp:
    return pd.to_datetime(df["ds"].min())


def get_max_data(df: pd.DataFrame) -> pd.Timestamp:
    return pd.to_datetime(df["ds"].max())
