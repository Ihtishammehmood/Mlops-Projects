import logging
import pickle

import pandas as pd
# from zenml import step



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

    encoder_path = "artifacts/feature_strategy.pkl"
    try:
        with open(encoder_path, "rb") as f:
            feature_strategy = pickle.load(f)
        logging.info(f"Loaded feature_strategy from {encoder_path}")
    except FileNotFoundError:
        logging.error(f"Feature strategy file not found at {encoder_path}")
        raise

    # Preprocess the data
    df = df.rename(columns={"remote_ratio": "work_status"})
    work_status_mapping = {0: "onsite", 50: "hybrid", 100: "remote"}
    df["work_status"] = df["work_status"].map(work_status_mapping)

    # Scale 'work_year' column
    df["work_year"] = year_scaler.transform(df[["work_year"]])

    features_to_encode = [
        "experience_level",
        "employment_type",
        "job_title",
        "employee_residence",
        "work_status",  # This is the renamed 'remote_ratio'
        "company_location",
        "company_size",
    ]

    # Transform the data using feature strategy
    df[features_to_encode] = feature_strategy.transform(df[features_to_encode])

    logging.info("Data preprocessing complete.")

    expected_columns = [
        "work_year",
        "experience_level",
        "employment_type",
        "job_title",
        "employee_residence",
        "work_status",
        "company_location",
        "company_size",
    ]
    return df[expected_columns]
