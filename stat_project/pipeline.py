import pandas as pd

# Functions for data: load, clean etc.
from .data_loader import (
    read_data,
    rename_columns,
    split_df,
    validate_content,
    validate_structure,
)
from .models.base_model import BaseModel
from .models.pmdarima import PmdarimaWrapper
from .models.prophet import ProphetWrapper
from .visualize import visualize_data

# Here are my models
MODEL_CATALOG = {
    "sarimax": PmdarimaWrapper,
    "prophet": ProphetWrapper,
}


class ForecastPipeline:
    """
    The main class
    """

    def __init__(self, path: str):
        self.path = path
        self.raw_df = None
        self.clean_df = None
        self.train_df = None
        self.test_df = None
        self.model: BaseModel = None
        self.forecast_df = None

    def prepare_data(self, split_date: str = "2025-01-01"):
        """
        Preparing data
        """
        print(f"Loading data from {self.path}...")
        self.raw_df = read_data(self.path)
        df_struct = validate_structure(self.raw_df)
        df_renamed = rename_columns(df_struct)
        self.clean_df = validate_content(df_renamed)

        self.train_df, self.test_df = split_df(self.clean_df, data=split_date)
        print("Data prepared.")
        return self

    def set_model(self, model_name: str, **params):
        """
        Get a model from MODEL_CATALOG
        """
        ModelClass = MODEL_CATALOG.get(model_name.lower())
        if not ModelClass:
            raise ValueError(f"""Model '{model_name}' not found.
                            Available: {list(MODEL_CATALOG.keys())}""")

        # Creating the model's object
        self.model = ModelClass(**params)
        print(f"Model set to '{model_name}' with params {params}.")
        return self

    def fit(self):
        """Fit a model"""
        if self.model is None:
            raise Exception("No model set. Call .set_model() first.")
        if self.train_df is None:
            raise Exception("Data not prepared. Call .prepare_data() first.")

        self.model.fit(self.train_df)
        return self

    def predict(self):
        """Making forecast"""
        if self.model is None or self.model.model is None:
            raise Exception("Model not fitted. Call .fit() first.")

        self.forecast_df = self.model.predict(self.test_df)
        print("Forecast generated.")
        return self

    def visualize(self):
        """Show vizualization"""
        visualize_data(self.forecast_df, self.train_df)

    def get_forecast(self) -> pd.DataFrame:
        """Return df with forecast"""
        return self.forecast_df
