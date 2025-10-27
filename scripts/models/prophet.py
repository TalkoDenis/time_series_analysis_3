import pandas as pd
from prophet import Prophet

from .base_model import BaseModel


class ProphetWrapper(BaseModel):
    """
    It's BaseModel for prophet
    """
    def fit(self, train_df: pd.DataFrame, y_col: str = 'y', exog_col: str = 'ds', seasonality_prior_scale: float = 25.0, country_name: str = 'US'):
        print("Starting prophet model...") 
        
        self.model = Prophet(
            seasonality_prior_scale=seasonality_prior_scale
        )
        model.add_country_holidays(country_name=country_name)
        model.fit(train_df)
        print("Prophet model fitted. Order:", self.model.order)

    def predict(self, test_df: pd.DataFrame, exog_col: str = 'ds') -> pd.DataFrame:
        print("Generating prophet forecast...")
        
        prediction_dates = test_df[['ds']]
        return self.model.predict(prediction_dates)