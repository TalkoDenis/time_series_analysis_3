def get_min_data(df) -> int:
    return df['ds'].min()


def get_max_data(df) -> int:
    return df['ds'].max()