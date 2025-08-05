from zenml import pipeline
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step

from pipelines.training_pipeline import ml_pipeline


@pipeline
def continuous_deployment_pipeline():
    """Run a training job and deploy an MLflow model deployment."""
    # Run the training pipeline
    trained_model = ml_pipeline()
    mlflow_model_deployer_step(workers=3, deploy_decision=True, model=trained_model)