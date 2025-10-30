import pandas as pd
from .get_min_max_data import get_max_data, get_min_data

def split_df(df: pd.DataFrame, data: str = "2025-01-01") -> tuple[pd.DataFrame, pd.DataFrame]:
    split_time = pd.to_datetime(data)
    
    if split_time < get_min_data(df):
        raise Exception(f"{data} is not valid! It is too small!")
    if split_time > get_max_data(df):
        raise Exception(f"{data} is not valid! It is too big!")

    mask = pd.to_datetime(df["ds"]) <= split_time
    train_df = df[mask]
    test_df = df[~mask]
    
    return train_df, test_df
