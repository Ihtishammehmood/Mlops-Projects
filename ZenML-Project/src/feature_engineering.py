import logging
from abc import ABC, abstractmethod

import category_encoders as ce
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

# Setup logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Abstract Base Class for Feature Engineering Strategy
# ----------------------------------------------------
class FeatureEngineeringStrategy(ABC):
    @abstractmethod
    def fit(self, df: pd.DataFrame):
        """
        Fits the strategy to the training data.
        For stateless transformations, this method might do nothing.

        Parameters:
        df (pd.DataFrame): The training dataframe to learn parameters from.
        """
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the learned transformation to the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe to transform.

        Returns:
        pd.DataFrame: A dataframe with the applied transformations.
        """
        pass


# Concrete Strategy for Log Transformation (Stateless)
# ----------------------------------------
class LogTransformation(FeatureEngineeringStrategy):
    def __init__(self, features):
        self.features = features

    def fit(self, df: pd.DataFrame):
        """Log transformation is stateless, so fit does nothing."""
        logging.info("LogTransformation is stateless. No fitting required.")
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Applying log transformation to features: {self.features}")
        df_transformed = df.copy()
        for feature in self.features:
            df_transformed[feature] = np.log1p(df[feature])
        logging.info("Log transformation completed.")
        return df_transformed


# Concrete Strategy for Standard Scaling
# --------------------------------------
class StandardScaling(FeatureEngineeringStrategy):
    def __init__(self, features):
        self.features = features
        self.scaler = StandardScaler()

    def fit(self, df: pd.DataFrame):
        logging.info(f"Fitting standard scaler to features: {self.features}")
        self.scaler.fit(df[self.features])
        logging.info("Standard scaler fitting completed.")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Applying standard scaling to features: {self.features}")
        df_transformed = df.copy()
        df_transformed[self.features] = self.scaler.transform(
            df_transformed[self.features]
        )
        logging.info("Standard scaling completed.")
        return df_transformed


# Concrete Strategy for Min-Max Scaling
# -------------------------------------
class MinMaxScaling(FeatureEngineeringStrategy):
    def __init__(self, features, feature_range=(0, 1)):
        self.features = features
        self.scaler = MinMaxScaler(feature_range=feature_range)

    def fit(self, df: pd.DataFrame):
        logging.info(f"Fitting Min-Max scaler to features: {self.features}")
        self.scaler.fit(df[self.features])
        logging.info("Min-Max scaler fitting completed.")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Applying Min-Max scaling to features: {self.features}")
        df_transformed = df.copy()
        df_transformed[self.features] = self.scaler.transform(
            df_transformed[self.features]
        )
        logging.info("Min-Max scaling completed.")
        return df_transformed


# Concrete Strategy for One-Hot Encoding
# --------------------------------------
class OneHotEncoding(FeatureEngineeringStrategy):
    def __init__(self, features):
        self.features = features
        self.encoder = OneHotEncoder(
            sparse=False, drop="first", handle_unknown="ignore"
        )

    def fit(self, df: pd.DataFrame):
        logging.info(f"Fitting one-hot encoder to features: {self.features}")
        self.encoder.fit(df[self.features])
        logging.info("One-hot encoder fitting completed.")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Applying one-hot encoding to features: {self.features}")
        df_transformed = df.copy()
        encoded_df = pd.DataFrame(
            self.encoder.transform(df[self.features]),
            columns=self.encoder.get_feature_names_out(self.features),
            index=df_transformed.index,
        )
        df_transformed = df_transformed.drop(columns=self.features)
        df_transformed = pd.concat([df_transformed, encoded_df], axis=1)
        logging.info("One-hot encoding completed.")
        return df_transformed


# concrete strategy for CatBoostEncoding
class CatBoostEncoding(FeatureEngineeringStrategy):
    def __init__(self, features, target: str):
        self.features = features
        self.target = target
        self.encoder = ce.CatBoostEncoder()

    def fit(self, df: pd.DataFrame):
        logging.info(f"Fitting CatBoost encoder to features: {self.features}")
        # CatBoostEncoder requires both X and y for fitting
        self.encoder.fit(df[self.features], df[self.target])
        logging.info("CatBoost encoder fitting completed.")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Applying CatBoost encoding to features: {self.features}")
        df_transformed = df.copy()
        df_transformed[self.features] = self.encoder.transform(
            df_transformed[self.features]
        )
        logging.info("CatBoost encoding completed.")
        return df_transformed
