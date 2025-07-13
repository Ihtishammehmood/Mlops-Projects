import logging
import pandas as pd
from zenml import step
from src.evaluation import R2Score, RMSE
from sklearn.base import RegressorMixin
from typing import Tuple
from typing_extensions import Annotated
import numpy as np


@step
def evaluate_model(
    model: RegressorMixin, 
    x_test: pd.DataFrame, 
    y_test: pd.Series
) -> Tuple[
    Annotated[float, "r2_score"],
    Annotated[float, "rmse"],
]:
    """
    Evaluates the model on the test data.
    Args:
        model: The trained model.
        x_test: The test data.
        y_test: The test labels.
    Returns:
        A tuple of r2_score and rmse.
    """
    try:
        prediction = model.predict(x_test)

        # The data was log-transformed in data_cleaning.py, so we need to inverse transform it.
        y_test_unlogged = np.expm1(y_test)
        prediction_unlogged = np.expm1(prediction)

        r2_class = R2Score()
        r2 = r2_class.calculate_score(y_test_unlogged, prediction_unlogged)
        logging.info(f"R2 Score: {r2}")

        rmse_class = RMSE()
        rmse = rmse_class.calculate_score(y_test_unlogged, prediction_unlogged)
        logging.info(f"RMSE: {rmse}")

        return r2, rmse
    except Exception as e:
        logging.error(f"Error in evaluating model: {e}")
        raise e