import pandas as pd
from prophet import Prophet

from .base_model import BaseModel


class ProphetWrapper(BaseModel):
    def __init__(self, **params):
        """
        It set params:'seasonality_prior_scale', 'country_name'.
        """
        super().__init__(**params)

    def fit(
        self, train_df: pd.DataFrame, y_col: str = "y", exog_col: str = "ds"
    ) -> None:
        """
        It learns the Prophet model.
        """
        print("Starting Prophet model...")

        seasonality_scale = self.params.get("seasonality_prior_scale", 25.0)
        country_name = self.params.get("country_name", "US")

        self.model = Prophet(seasonality_prior_scale=seasonality_scale)

        if country_name:
            self.model.add_country_holidays(country_name=country_name)

        self.model.fit(train_df)

        print("Prophet model fitted.")

    def predict(
        self, test_df: pd.DataFrame, exog_col: str = "ds"
    ) -> pd.DataFrame:
        """
        It makes forecast
        """
        print("Generating Prophet forecast...")

        prediction_dates = test_df[["ds"]]
        raw_forecast = self.model.predict(prediction_dates)

        result_df = pd.merge(
            test_df,
            raw_forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]],
            on="ds",
        )

        result_df = result_df.rename(
            columns={
                "y": "Real_data",
                "yhat": "Forecust_data",
                "yhat_lower": "min",
                "yhat_upper": "max",
            }
        )

        result_df["Diff"] = result_df["Real_data"] - result_df["Forecust_data"]

        final_cols = ["ds", "Real_data", "Forecust_data", "min", "max", "Diff"]
        return result_df[final_cols].reset_index(drop=True)
