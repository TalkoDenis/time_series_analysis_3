from prophet import Prophet
from scripts.is_valid_country_name import is_valid_country_name
from scripts.is_valid_seasonality_prior_scale import is_valid_seasonality_prior_scale
import pandas as pd
from .base_model import BaseModel

class ProphetWrapper(BaseModel):
    """
    It's BaseModel for prophet
    """
    def fit(self, train_df: pd.DataFrame, y_col: str = 'y', exog_col: str = 'ds'):
        print("Starting prophet model...")
        
        m = self.params.get('m', 12) 
        
        self.model = pm.auto_arima(
            train_df[[y_col]],
            exogenous=train_df[[exog_col]],
            m=m,
            start_p=1, start_q=1,
            test='adf',
            max_p=3, max_q=3,
            start_P=0, seasonal=True,
            d=None, D=1,
            trace=False,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True
        )
        print("Pmdarima model fitted. Order:", self.model.order)

    def predict(self, test_df: pd.DataFrame, exog_col: str = 'ds') -> pd.DataFrame:
        print("Generating pmdarima forecast...")
        
        n_periods = len(test_df)
        exogenous_data = test_df[[exog_col]]
        
        fitted, confint = self.model.predict(
            n_periods=n_periods,
            exogenous=exogenous_data,
            return_conf_int=True
        )
        
        result_df = test_df.copy()
        result_df['Forecust_data'] = fitted
        result_df['min'] = confint[:, 0]
        result_df['max'] = confint[:, 1]
        result_df = result_df.rename(columns={'y': 'Real_data'})
        result_df['Diff'] = result_df['Real_data'] - result_df['Forecust_data']
        
        return result_df.reset_index(drop=True)