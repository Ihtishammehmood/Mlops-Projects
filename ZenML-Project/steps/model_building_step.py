import logging
from typing import Annotated
import joblib

import mlflow
import pandas as pd
# from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from zenml import ArtifactConfig, Model, step
from zenml.client import Client
from zenml.enums import ArtifactType

# Get the active experiment tracker from ZenML
experiment_tracker = Client().active_stack.experiment_tracker

model = Model(
    name="salary_predictor",
    version=None,
    license="Apache 2.0",
    description="A Salary Prediction Model.",
)


@step(enable_cache=False, experiment_tracker=experiment_tracker.name, model=model)
def model_building_step(
    X_train: pd.DataFrame, y_train: pd.Series
) -> Annotated[
    Pipeline, ArtifactConfig(name="sklearn_pipeline", artifact_type=ArtifactType.MODEL)
]:
    """
    Builds and trains a Linear Regression model using scikit-learn wrapped in a pipeline.

    Parameters:
    X_train (pd.DataFrame): The training data features.
    y_train (pd.Series): The training data labels/target.

    Returns:
    Pipeline: The trained scikit-learn pipeline including preprocessing and the Linear Regression model.
    """
    # Ensure the inputs are of the correct type
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame.")
    if not isinstance(y_train, pd.Series):
        raise TypeError("y_train must be a pandas Series.")

    # Identify categorical and numerical columns
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns
    numerical_cols = X_train.select_dtypes(exclude=["object", "category"]).columns

    logging.info(f"Categorical columns: {categorical_cols.tolist()}")
    logging.info(f"Numerical columns: {numerical_cols.tolist()}")

    # lr = LinearRegression()
    rf = RandomForestRegressor(n_estimators=200,
                               max_depth = 15,
                               min_samples_split = 5,
                               min_samples_leaf = 2,
                               max_features = 'sqrt',
                               bootstrap = True,
                               n_jobs = -1,
                               random_state=42)

    # Start an MLflow run to log the model training process
    if not mlflow.active_run():
        mlflow.start_run()  # Start a new MLflow run if there isn't one active

    try:
        # Enable autologging for scikit-learn to automatically capture model metrics, parameters, and artifacts
        mlflow.sklearn.autolog()

        logging.info("Building and training the model.")
        rf.fit(X_train, y_train)
        logging.info("Model training completed.")

        # Log the columns that the model expects
        expected_columns = list(X_train.columns)
        logging.info(f"Model expects the following columns: {expected_columns}")

    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise e

    finally:
        # End the MLflow run
        mlflow.end_run()

    pipeline = Pipeline(steps=[("model", rf)])
    joblib.dump(pipeline, "artifacts/model_trainer.pkl")

    return pipeline
