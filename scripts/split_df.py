def split_df(df, data='2025-01-01'):
    if pd.to_datetime(data) < get_min_data(df):
        raise Exception(f'{data} is not valid! It is too small!')
    if pd.to_datetime(data) > get_max_data(df):
        raise Exception(f'{data} is not valid! It is too big!')
    train_df = df[df['ds'] <= pd.to_datetime(data)]
    future_df = df[df['ds'] > pd.to_datetime(data)]
    return train_df, future_df