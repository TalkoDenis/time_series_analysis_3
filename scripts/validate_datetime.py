def validate_datatime(df):
    try:
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
    except ValueError as e:
        print("Some values couldn’t be parsed as dates.")
        print(e)
    except Exception as e:
        print(f"Failed to convert to datetime: {e}")
    else:
        return df.sort_values(by=df.columns[0])