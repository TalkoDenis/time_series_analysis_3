import pandas as pd

from scripts.get_min_max_data import get_max_data, get_min_data


def split_df_get_train(
    df: pd.DataFrame, data: str = "2025-01-01"
) -> pd.DataFrame:
    if pd.to_datetime(data) < get_min_data(df):
        raise Exception(f"{data} is not valid! It is too small!")
    if pd.to_datetime(data) > get_max_data(df):
        raise Exception(f"{data} is not valid! It is too big!")
    return df[pd.to_datetime(df["ds"]) <= pd.to_datetime(data)]


def split_df_get_test(
    df: pd.DataFrame, data: str = "2025-01-01"
) -> pd.DataFrame:
    if pd.to_datetime(data) < get_min_data(df):
        raise Exception(f"{data} is not valid! It is too small!")
    if pd.to_datetime(data) > get_max_data(df):
        raise Exception(f"{data} is not valid! It is too big!")
    return df[pd.to_datetime(df["ds"]) > pd.to_datetime(data)]
