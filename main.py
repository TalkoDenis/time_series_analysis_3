from scripts.get_forecast import get_forecast
from scripts.learning_model import learning_model
from scripts.read_data import read_data
from scripts.rename_columns import rename_columns
from scripts.split_df import split_df
from scripts.validate_date import validate_data
from scripts.visualize_data import visualize_data

df = read_data("./data/data.csv")
df = validate_data(df)
df_prophet = rename_columns(df)
del df
train_df, test_df = split_df(df_prophet)

model = learning_model(train_df, train_param="y", exogenous_param="ds")
forecast_df = get_forecast(model, train_df, test_df)

visualize_data(forecast_df, train_df)
