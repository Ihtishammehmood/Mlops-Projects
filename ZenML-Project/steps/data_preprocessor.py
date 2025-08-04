# steps/data_preprocessor.py
import pandas as pd
import pickle
import logging
# import json # No longer needed
from zenml import step
import numpy as np
@step
def data_preprocessor(df: pd.DataFrame) -> pd.DataFrame:
    """
    Loads the fitted scaler and encoder, and preprocesses the input data.
    Returns a DataFrame with the required column names.
    """
    logging.basicConfig(level=logging.INFO)
    logging.info("Loading fitted scaler and encoder...")

    # Load the fitted scaler
    scaler_path = "artifacts/year_scaler.pkl"
    try:
        with open(scaler_path, "rb") as f:
            year_scaler = pickle.load(f)
        logging.info(f"Loaded year_scaler from {scaler_path}")
    except FileNotFoundError:
        logging.error(f"Scaler file not found at {scaler_path}")
        raise

    # Load the fitted feature strategy
    encoder_path = "artifacts/feature_strategy.pkl"
    try:
        with open(encoder_path, "rb") as f:
            feature_strategy = pickle.load(f)
        logging.info(f"Loaded feature_strategy from {encoder_path}")
    except FileNotFoundError:
        logging.error(f"Feature strategy file not found at {encoder_path}")
        raise

    # Load the data from json
    # df = pd.read_json(json_data, orient="split") # Take DF as input instead

    # Preprocess the data
    df = df.rename(columns={'remote_ratio': 'work_status'})
    work_status_mapping = {
        0: 'onsite',
        50: 'hybrid',
        100: 'remote'
    }
    df['work_status'] = df['work_status'].map(work_status_mapping)

    # Scale 'work_year' column
    df['work_year'] = year_scaler.transform(df[['work_year']])

    features_to_encode = [
        'experience_level',
        'employment_type',
        'job_title',
        'employee_residence',
        'work_status',  # This is the renamed 'remote_ratio'
        'company_location',
        'company_size'
    ]

    # Transform the data using feature strategy
    df[features_to_encode] = feature_strategy.transform(df[features_to_encode])

    # convert the data back to json format
    # df_transformed = df.to_json(orient="split") # No longer needed

    logging.info("Data preprocessing complete.")
    # Ensure columns are in the expected order
    expected_columns = [
        'work_year',
        'experience_level',
        'employment_type',
        'job_title',
        'employee_residence',
        'work_status',
        'company_location',
        'company_size'
    ]
    return df[expected_columns]