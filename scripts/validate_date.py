import pandas as pd
from pandas.errors import OutOfBoundsDatetime


def validate_data(df):
    try:
        df.iloc[:, 1] = pd.to_numeric(df.iloc[:, 1], downcast='integer')
    except OutOfBoundsDatetime as e:
        print("DValues are out of range (too large/small). Setting invalid to NaT.")
        print(e)
    except TypeError as e:
        print("Invalid type for column.")
        print(e)
    else:
        return df.sort_values(by=df.columns[0])