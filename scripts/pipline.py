import pandas as pd
from .models.base_model import BaseModel
from .models.pmdarima import PmdarimaWrapper
from .models.prophet import ProphetWrapper 

# Импортируй свои функции подготовки данных
from .data_loader import (
    read_data, 
    validate_structure, 
    rename_columns, 
    validate_content, 
    split_df
)
from .visualize import visualize_data

# "Фабрика" моделей. Здесь ты регистрируешь все свои модели.
MODEL_CATALOG = {
    "sarimax": PmdarimaWrapper,
    # "prophet": ProphetWrapper, # (когда добавишь)
}

class ForecastPipeline:
    """
    Главный класс-оркестратор. 
    Хранит состояние (данные) и управляет моделями.
    """
    def __init__(self, path: str):
        self.path = path
        self.raw_df = None
        self.clean_df = None
        self.train_df = None
        self.test_df = None
        self.model: BaseModel = None # Подсказка типа
        self.forecast_df = None

    def prepare_data(self, split_date: str = '2025-01-01'):
        """
        Выполняет всю цепочку подготовки данных.
        """
        print(f"Loading data from {self.path}...")
        self.raw_df = read_data(self.path)
        df_struct = validate_structure(self.raw_df)
        df_renamed = rename_columns(df_struct)
        self.clean_df = validate_content(df_renamed)
        
        self.train_df, self.test_df = split_df(self.clean_df, data=split_date)
        print("Data prepared.")
        return self # Возвращаем self для "цепочки" вызовов

    def set_model(self, model_name: str, **params):
        """
        Выбирает и инициализирует модель из каталога.
        """
        ModelClass = MODEL_CATALOG.get(model_name.lower())
        if not ModelClass:
            raise ValueError(f"Model '{model_name}' not found. Available: {list(MODEL_CATALOG.keys())}")
        
        # Создаем ОБЪЕКТ модели с ее гиперпараметрами
        self.model = ModelClass(**params) 
        print(f"Model set to '{model_name}' with params {params}.")
        return self

    def fit(self):
        """Обучает выбранную модель."""
        if self.model is None:
            raise Exception("No model set. Call .set_model() first.")
        if self.train_df is None:
            raise Exception("Data not prepared. Call .prepare_data() first.")
            
        self.model.fit(self.train_df)
        return self

    def predict(self):
        """Генерирует прогноз."""
        if self.model is None or self.model.model is None:
            raise Exception("Model not fitted. Call .fit() first.")
            
        self.forecast_df = self.model.predict(self.test_df)
        print("Forecast generated.")
        return self

    def visualize(self):
        """Показывает график."""
        visualize_data(self.forecast_df, self.train_df)
        
    def get_forecast(self) -> pd.DataFrame:
        """Возвращает итоговый DataFrame с прогнозом."""
        return self.forecast_df