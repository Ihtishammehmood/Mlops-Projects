import pandas as pd
from zenml import step
import pickle
# import os
import logging

@step
def data_preprocessor(json_data: str) -> pd.DataFrame:
    """
    Loads the fitted scaler and encoder, and preprocesses the input data.
    """
    logging.info("Loading fitted scaler and encoder...")

    # Load the fitted scaler
    scaler_path = "artifacts/year_scaler.pkl"
    with open(scaler_path, "rb") as f:
        year_scaler = pickle.load(f)
    logging.info(f"Loaded year_scaler from {scaler_path}")

    # Load the fitted feature strategy
    encoder_path = "artifacts/feature_strategy.pkl"
    with open(encoder_path, "rb") as f:
        feature_strategy = pickle.load(f)
    logging.info(f"Loaded feature_strategy from {encoder_path}")

    # Load the data from json
    df = pd.read_json(json_data, orient="split")
    
    # expected_columns = [
    #     'work_year',
    #     'experience_level',
    #     'employment_type',
    #     'job_title',
    #     'salary_in_usd',
    #     'employee_residence',
    #     'work_status',
    #     'company_location',
    #     'company_size']
    
    # df= [[expected_columns]]

    # Preprocess the data
    df = df.rename(columns={'remote_ratio': 'work_status'})
    work_status_mapping = {
        0: 'onsite',
        50: 'hybrid',
        100: 'remote'
    }
    df['work_status'] = df['work_status'].map(work_status_mapping)

    # Scale 'work_year' column
    df_scaled = year_scaler.transform(df)

    # Transform the data using feature strategy
    df_transformed = feature_strategy.transform(df_scaled)
    
    # convert the data back to json format
    # df_transformed = df_transformed.to_json(orient="split")

    logging.info("Data preprocessing complete.")
    return df_transformed
