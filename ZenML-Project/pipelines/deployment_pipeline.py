from pipelines.training_pipeline import ml_pipeline
# from steps.dynamic_importer import dynamic_importer
# from steps.model_loader import model_loader
# from steps.prediction_service_loader import prediction_service_loader
# from steps.data_preprocessor import data_preprocessor
# from steps.predictor import predictor
from zenml import pipeline
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step



@pipeline
def continuous_deployment_pipeline():
    """Run a training job and deploy an MLflow model deployment."""
    # Run the training pipeline
    trained_model = ml_pipeline()  # No need for is_promoted return value anymore
    # trained_model = model_loader('salary_predictor')
    
    # (Re)deploy the trained model
    mlflow_model_deployer_step(workers=3, deploy_decision=True, model=trained_model)