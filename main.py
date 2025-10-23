from scripts.learning_model import learning_model
from scripts.read_data import read_data
from scripts.rename_columns import rename_columns
from scripts.save_result_df import save_result_df
from scripts.split_df import split_df
from scripts.validate_date import validate_data
from scripts.validate_datetime import validate_datatime
from scripts.visualize_data import visualize_data
from scripts.get_forecast import get_forecast


df = read_data('./data/data.csv')
df = validate_data(df)
df_prophet = rename_columns(df)
del df
train_df, future_df = split_df(df_prophet)

learning_model(train_df, train_param='y', exogenous_param='ds')
forecast_df = get_forecast

visualize_data(forecast_df)
# print(forecast.head())