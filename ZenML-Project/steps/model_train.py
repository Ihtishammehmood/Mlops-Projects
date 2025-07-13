import logging
import mlflow
from zenml.client import Client
import pandas as pd
from zenml import step, ArtifactConfig
from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from typing_extensions import Annotated
from zenml import Model
from zenml.enums import ArtifactType


model = Model(
    name="Salary_Predictor",
    version=None,
    license="Apache 2.0",
    description="Predicts the salaries for all data science and ai related jobs.",
)


experiment_tracker = Client().active_stack.experiment_tracker

@step(enable_cache=False, experiment_tracker=experiment_tracker.name, model=model)
def train_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str = "LinearRegression",
) -> Annotated[RegressorMixin, ArtifactConfig(name="trained_model", artifact_type = ArtifactType.MODEL)]:
    """
    Args:
        x_train: pd.DataFrame
        y_train: pd.Series
        model_name: Name of the model to train.
    Returns:
        model: RegressorMixin
    """
    try:
        model = None

        if model_name.lower() == "linearregression":
            mlflow.sklearn.autolog()
            model = LinearRegressionModel()
            trained_model = model.train(x_train, y_train)
            return trained_model
        else:
            raise ValueError(f"Model {model_name} not supported")



    except Exception as e:
        logging.error(e)
        raise e