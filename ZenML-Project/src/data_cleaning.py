import logging
from abc import ABC, abstractmethod
from typing import Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from category_encoders import CatBoostEncoder

class DataStrategy(ABC):
    """Abstract class defining strategy for handling data"""

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
        pass


class DataPreprocessStrategy(DataStrategy):
    """
    Preprocesses the data by dropping unnecessary columns and filling missing values.
    """

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            df = data.copy()

            # Drop columns not needed
            drop_cols = ["salary", "salary_currency"]
            df = df.drop(columns=[col for col in drop_cols if col in df.columns])

            # Example: Fill missing values (if required)
            # df.fillna(df.median(numeric_only=True), inplace=True)

            return df

        except Exception as e:
            logging.error(f"DataPreprocessStrategy error: {e}")
            raise


class DataDivideStrategy(DataStrategy):
    """
    Divides data into training and testing sets, encodes categorical features, and scales numeric ones.
    """

    def handle_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        try:
            df = data.copy()

            # Separate features and target
            if "salary_in_usd" not in df.columns:
                raise ValueError("Target column 'salary_in_usd' not found in data.")

            X = df.drop(columns="salary_in_usd")
            y = df["salary_in_usd"]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Encode categorical features using CatBoostEncoder
            cat_features = X_train.select_dtypes(include='object').columns.tolist()
            encoder = CatBoostEncoder(cols=cat_features, drop_invariant=True)
            X_train = encoder.fit_transform(X_train, y_train)
            X_test = encoder.transform(X_test)

            # Scale numeric features
            num_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
            scaler = StandardScaler()

            X_train[num_features] = scaler.fit_transform(X_train[num_features])
            X_test[num_features] = scaler.transform(X_test[num_features])

            # Log-transform the target
            y_train = np.log1p(y_train)
            y_test = np.log1p(y_test)

            return X_train, X_test, y_train, y_test

        except Exception as e:
            logging.error(f"DataDivideStrategy error: {e}")
            raise


class DataCleaning:
    """
    Applies a data cleaning strategy to the dataset.
    """

    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        self.df = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
        return self.strategy.handle_data(self.df)
