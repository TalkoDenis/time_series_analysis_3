


df = read_data('./data/data.csv')
df = validate_data(df)
df_prophet = rename_columns(df)
del df
train_df, future_df = split_df(df_prophet)
@timing
forecast = model_learning(train_df, future_df, seasonality_prior_scale=25.0, country_name='US')
visualize_data(df_prophet, forecast)