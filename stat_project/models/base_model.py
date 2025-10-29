from abc import ABC, abstractmethod

import pandas as pd


class BaseModel(ABC):
    """
    It's just base model class.
    """

    def __init__(self, **params):
        """
        It saves params
        """
        self.params = params
        self.model = None

    @abstractmethod
    def fit(
        self, train_df: pd.DataFrame, y_col: str = "y", exog_col: str = "ds"
    ):
        """
        It learns a model
        """
        pass

    @abstractmethod
    def predict(
        self, test_df: pd.DataFrame, exog_col: str = "ds"
    ) -> pd.DataFrame:
        """
        It makes predict
        """
        pass
