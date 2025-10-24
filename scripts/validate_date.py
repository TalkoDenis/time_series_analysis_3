import pandas as pd
from pandas.errors import OutOfBoundsDatetime

def validate_data(df):
    if df.empty:
        raise Exception("DataFrame is empty!")

    required_cols = {'date', 'value'}
    
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise Exception(f"DataFrame is not valid! Missing columns: {missing}")

    try:
        df['value'] = pd.to_numeric(df['value'], downcast='integer')
    except (OutOfBoundsDatetime, TypeError, ValueError) as e:
        print(f"Failed to convert 'value' column to numeric: {e}")
        raise Exception("Invalid data type in 'value' column.")
    
    return df.sort_values(by='date')