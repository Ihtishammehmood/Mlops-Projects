import click
from piplines.training_pipline import train_pipeline
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri


@click.command()
def main():
    """
    Run the ML pipeline and start the MLflow UI for experiment tracking.
    """
    # Run the pipeline
    train_pipeline(data_path="data/salaries.csv")
    print(
        "Now run \n "
        f"    mlflow ui --backend-store-uri '{get_tracking_uri()}'\n"
        "To inspect your experiment runs within the mlflow UI.\n"
        "You can find your runs tracked within the experiment."
    )


if __name__ == "__main__":
    main()
    # train_pipeline(data_path="data/salaries.csv")