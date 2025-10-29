import pandas as pd


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={df.columns[0]: "ds", df.columns[1]: "y"})
