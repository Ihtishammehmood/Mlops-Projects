import logging
import os
import pickle

import numpy as np
import pandas as pd
from zenml import step

from src.feature_engineering import CatBoostEncoding, MinMaxScaling


@step
def feature_engineering_step(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Feature engineering step that fits on training data and transforms both
    training and testing data to prevent data leakage.
    """
    # dropping unnecessary columns
    drop_col = ["salary_currency", "salary"]
    X_train.drop(drop_col, axis=1, inplace=True)
    X_test.drop(drop_col, axis=1, inplace=True)

    y_train = y_train.apply(lambda x: np.log(x))
    y_test = y_test.apply(lambda x: np.log(x))

    X_train = X_train.rename(columns={"remote_ratio": "work_status"})
    X_test = X_test.rename(columns={"remote_ratio": "work_status"})

    work_status_mapping = {0: "onsite", 50: "hybrid", 100: "remote"}
    X_train["work_status"] = X_train["work_status"].map(work_status_mapping)
    X_test["work_status"] = X_test["work_status"].map(work_status_mapping)

    year_scaler = MinMaxScaling(features=["work_year"], feature_range=(0, 1))
    year_scaler.fit(X_train)
    X_train_scaled = year_scaler.transform(X_train)
    X_test_scaled = year_scaler.transform(X_test)

    features_to_encode = [
        "experience_level",
        "employment_type",
        "job_title",
        "employee_residence",
        "work_status",  # This is the renamed 'remote_ratio'
        "company_location",
        "company_size",
    ]
    feature_strategy = CatBoostEncoding(
        features=features_to_encode, target="salary_in_usd"
    )

    logging.info("Fitting feature engineering strategy on training data.")
    # For CatBoostEncoding, the target column is needed for fitting.
    # We'll add it to the training data temporarily for the fit method.
    X_train_for_fit = X_train_scaled.copy()
    X_train_for_fit["salary_in_usd"] = y_train.values
    feature_strategy.fit(X_train_for_fit)

    artifacts_dir = "artifacts"
    if not os.path.exists(artifacts_dir):
        os.makedirs(artifacts_dir)

    # Save the fitted scaler
    year_scaler_path = os.path.join(artifacts_dir, "year_scaler.pkl")
    with open(year_scaler_path, "wb") as f:
        pickle.dump(year_scaler, f)
    logging.info(f"Saved year_scaler to {year_scaler_path}")

    # Save the fitted feature strategy
    feature_strategy_path = os.path.join(artifacts_dir, "feature_strategy.pkl")
    with open(feature_strategy_path, "wb") as f:
        pickle.dump(feature_strategy, f)
    logging.info(f"Saved feature_strategy to {feature_strategy_path}")

    logging.info("Transforming training and testing data.")
    X_train_transformed = feature_strategy.transform(X_train_scaled)
    X_test_transformed = feature_strategy.transform(X_test_scaled)

    return X_train_transformed, X_test_transformed
