# -*- coding: utf-8 -*-

# -- Sheet --

from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
%matplotlib inline
from matplotlib.pylab import rcParams

from statsmodels.tsa.stattools import adfuller
!pip install pmdarima -q
import pmdarima as pm
from statsmodels.tsa.seasonal import seasonal_decompose

df = pd.read_csv("https://raw.githubusercontent.com/AileenNielsen/TimeSeriesAnalysisWithPython/master/data/AirPassengers.csv")

df.head()

df['Month'] = pd.to_datetime(df['Month'], infer_datetime_format=True)
df = df.set_index(['Month'])

df['#Passengers_diff'] = df['#Passengers'].diff(periods=12)
df.info()

df['#Passengers_diff'].fillna(method='backfill', inplace=True)

result = seasonal_decompose(df['#Passengers'], model='multiplicative', period=12)
trend = result.trend.dropna()
seasonal = result.seasonal.dropna()
residual = result.resid.dropna()

# Plot the decomposed components
plt.figure(figsize=(6,6))

plt.subplot(4, 1, 1)
plt.plot(df['#Passengers'], label='Original Series')
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(trend, label='Trend')
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(seasonal, label='Seasonal')
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(residual, label='Residuals')
plt.legend()

plt.tight_layout()
plt.show()

df['month_index'] = df.index.month

SARIMAX_model = pm.auto_arima(df[['#Passengers']], exogenous=df[['month_index']],
                           start_p=1, start_q=1,
                           test='adf',
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=None, D=1,
                           trace=False,
                           error_action='ignore',
                           suppress_warnings=True,
                           stepwise=True)

def sarimax_forecast(SARIMAX_model, periods=24):
    # Forecast
    n_periods = periods

    forecast_df = pd.DataFrame({"month_index": pd.date_range(df.index[-1], periods=n_periods, freq='MS').month},
                               index=pd.date_range(df.index[-1] + pd.DateOffset(months=1), periods=n_periods, freq='MS'))

    fitted, confint = SARIMAX_model.predict(n_periods=n_periods,
                                            return_conf_int=True,
                                            exogenous=forecast_df[['month_index']])
    index_of_fc = pd.date_range(df.index[-1] + pd.DateOffset(months=1), periods=n_periods, freq='MS')

    # make series for plotting purpose
    fitted_series = pd.Series(fitted, index=index_of_fc)
    lower_series = pd.Series(confint[:, 0], index=index_of_fc)
    upper_series = pd.Series(confint[:, 1], index=index_of_fc)

    # Plot
    plt.figure(figsize=(15, 7))
    plt.plot(df["#Passengers"], color='#1f76b4')
    plt.plot(fitted_series, color='darkgreen')
    plt.fill_between(lower_series.index,
                     lower_series,
                     upper_series,
                     color='k', alpha=.15)

    plt.title("SARIMAX - Forecast of Airline Passengers")
    plt.show()

sarimax_forecast(SARIMAX_model, periods=24)

SARIMAX_model

# -- Sheet Duplicate --

# from datetime import datetime
# import numpy as np
# import pandas as pd
# import matplotlib.pylab as plt
# %matplotlib inline
# from matplotlib.pylab import rcParams

# from statsmodels.tsa.stattools import adfuller
# !pip install pmdarima -q
# import pmdarima as pm
# from statsmodels.tsa.seasonal import seasonal_decompose

import pandas as pd
import pmdarima as pm
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("/data/notebook_files/metrics.csv")
df.head()

df = df.groupby('week').agg({'fd_cnt':'sum'}).reset_index()
df.head()





df['week'] = pd.to_datetime(df['week'], infer_datetime_format=True)
df = df.set_index(['week'])

df['fd_cnt_diff'] = df['fd_cnt'].diff(periods=12)
df.info()

df['fd_cnt_diff'].fillna(method='backfill', inplace=True)

result = seasonal_decompose(df['fd_cnt'], model='multiplicative', period=12)
trend = result.trend.dropna()
seasonal = result.seasonal.dropna()
residual = result.resid.dropna()

# Plot the decomposed components
plt.figure(figsize=(6,6))

plt.subplot(4, 1, 1)
plt.plot(df['fd_cnt'], label='Original Series')
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(trend, label='Trend')
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(seasonal, label='Seasonal')
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(residual, label='Residuals')
plt.legend()

plt.tight_layout()
plt.show()

df.head()



# df['week_index'] = df[df.index < '2025-05-01'].index.to_series().dt.isocalendar().week

# SARIMAX_model = pm.auto_arima(df[['fd_cnt']], exogenous=df[['week_index']],
#                            start_p=1, start_q=1,
#                            test='adf',
#                            max_p=3, max_q=3, m=12,
#                            start_P=0, seasonal=True,
#                            d=None, D=1,
#                            trace=False,
#                            error_action='ignore',
#                            suppress_warnings=True,
#                            stepwise=True)

# def sarimax_forecast(SARIMAX_model, periods=1):
#     # Forecast
#     n_periods = periods

#     forecast_df = pd.DataFrame({"week_index": pd.date_range(df.index[-1], periods=n_periods, freq='MS').to_series().dt.isocalendar().week},
#                                index=pd.date_range(df.index[-1] + pd.DateOffset(months=1), periods=n_periods, freq='MS'))

#     fitted, confint = SARIMAX_model.predict(n_periods=n_periods,
#                                             return_conf_int=True,
#                                             exogenous=forecast_df[['week_index']])
#     index_of_fc = pd.date_range(df.index[-1] + pd.DateOffset(months=1), periods=n_periods, freq='MS')

#     # make series for plotting purpose
#     fitted_series = pd.Series(fitted, index=index_of_fc)
#     lower_series = pd.Series(confint[:, 0], index=index_of_fc)
#     upper_series = pd.Series(confint[:, 1], index=index_of_fc)

#     # Plot
#     plt.figure(figsize=(15, 7))
#     plt.plot(df["fd_cnt"], color='#1f76b4')
#     plt.plot(fitted_series, color='darkgreen')
#     plt.fill_between(lower_series.index,
#                      lower_series,
#                      upper_series,
#                      color='k', alpha=.15)

#     plt.title("SARIMAX - Forecast of Airline Passengers")
#     plt.show()

# sarimax_forecast(SARIMAX_model, periods=3)

# SARIMAX_model

df['week_index'] = df.index.to_series().dt.isocalendar().week

train = df[df.index < '2025-05-01']
test = df[df.index >= '2025-05-01']

print("Данные для обучения (train):")
print(train.tail())
print("\nДанные для теста (test):")
print(test.head())


print("\nОбучение модели...")
SARIMAX_model = pm.auto_arima(train[['fd_cnt']], exogenous=train[['week_index']],
                           start_p=1, start_q=1,
                           test='adf',
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=None, D=1,
                           trace=False,
                           error_action='ignore',
                           suppress_warnings=True,
                           stepwise=True)

print("Модель обучена. Параметры:", SARIMAX_model.order, SARIMAX_model.seasonal_order)


# periods больше не нужен как аргумент, мы точно знаем, на сколько предсказывать
def forecast_and_plot(model, train_data, test_data):
    # Определяем, на сколько периодов вперед нужно сделать прогноз
    n_periods = len(test_data)

    # Создаем DataFrame с будущими экзогенными переменными.
    # Индекс берем из тестовых данных, чтобы все точно совпало!
    forecast_exog = test_data[['week_index']]
    
    # Делаем прогноз
    fitted, confint = model.predict(n_periods=n_periods,
                                    return_conf_int=True,
                                    exogenous=forecast_exog)
    
    # Создаем pd.Series с прогнозом, используя индекс тестовых данных
    fitted_series = pd.Series(fitted, index=test_data.index)
    lower_series = pd.Series(confint[:, 0], index=test_data.index)
    upper_series = pd.Series(confint[:, 1], index=test_data.index)

    # Plot
    plt.figure(figsize=(15, 7))
    # Рисуем данные, на которых модель обучалась
    plt.plot(train_data['fd_cnt'], color='blue', label='Обучающие данные (Train)')
    # Рисуем реальные данные, которые модель не видела
    plt.plot(test_data['fd_cnt'], color='orange', label='Реальные данные (Test Fact)')
    # Рисуем прогноз модели
    plt.plot(fitted_series, color='darkgreen', label='Прогноз (Forecast)')
    
    # Рисуем доверительный интервал
    plt.fill_between(lower_series.index,
                     lower_series,
                     upper_series,
                     color='k', alpha=.15)

    plt.title("SARIMAX - Прогноз и сравнение с фактом")
    plt.legend()
    plt.show()

    # Возвращаем таблицу для анализа
    result_df = pd.DataFrame({
        'Факт': test_data['fd_cnt'],
        'Прогноз': fitted_series
    })
    result_df['Разница'] = result_df['Факт'] - result_df['Прогноз']
    return result_df

# --- ШАГ 4: Вызов функции и просмотр результатов ---
result_table = forecast_and_plot(SARIMAX_model, train, test)

print("\nТаблица с результатами:")
print(result_table)

# -- Sheet 3 --

import pandas as pd
import pmdarima as pm
import matplotlib.pyplot as plt
import numpy as np


df['week_index'] = df.index.to_series().dt.isocalendar().week

train = df[df.index < '2025-05-01']
test = df[df.index >= '2025-05-01']

print("Данные для обучения (train):")
print(train.tail())
print("\nДанные для теста (test):")
print(test.head())


print("\nОбучение модели...")
SARIMAX_model = pm.auto_arima(train[['fd_cnt']], exogenous=train[['week_index']],
                           start_p=1, start_q=1,
                           test='adf',
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=None, D=1,
                           trace=False,
                           error_action='ignore',
                           suppress_warnings=True,
                           stepwise=True)

print("Модель обучена. Параметры:", SARIMAX_model.order, SARIMAX_model.seasonal_order)


# periods больше не нужен как аргумент, мы точно знаем, на сколько предсказывать
def forecast_and_plot(model, train_data, test_data):
    # Определяем, на сколько периодов вперед нужно сделать прогноз
    n_periods = len(test_data)

    # Создаем DataFrame с будущими экзогенными переменными.
    # Индекс берем из тестовых данных, чтобы все точно совпало!
    forecast_exog = test_data[['week_index']]
    
    # Делаем прогноз
    fitted, confint = model.predict(n_periods=n_periods,
                                    return_conf_int=True,
                                    exogenous=forecast_exog)
    
    # Создаем pd.Series с прогнозом, используя индекс тестовых данных
    fitted_series = pd.Series(fitted, index=test_data.index)
    lower_series = pd.Series(confint[:, 0], index=test_data.index)
    upper_series = pd.Series(confint[:, 1], index=test_data.index)

    # Plot
    plt.figure(figsize=(15, 7))
    # Рисуем данные, на которых модель обучалась
    plt.plot(train_data['fd_cnt'], color='blue', label='Обучающие данные (Train)')
    # Рисуем реальные данные, которые модель не видела
    plt.plot(test_data['fd_cnt'], color='orange', label='Реальные данные (Test Fact)')
    # Рисуем прогноз модели
    plt.plot(fitted_series, color='darkgreen', label='Прогноз (Forecast)')
    
    # Рисуем доверительный интервал
    plt.fill_between(lower_series.index,
                     lower_series,
                     upper_series,
                     color='k', alpha=.15)

    plt.title("SARIMAX - Прогноз и сравнение с фактом")
    plt.legend()
    plt.show()

    # Возвращаем таблицу для анализа
    result_df = pd.DataFrame({
        'Факт': test_data['fd_cnt'],
        'Прогноз': fitted_series
    })
    result_df['Разница'] = result_df['Факт'] - result_df['Прогноз']
    return result_df

# --- ШАГ 4: Вызов функции и просмотр результатов ---
result_table = forecast_and_plot(SARIMAX_model, train, test)

print("\nТаблица с результатами:")
print(result_table)

