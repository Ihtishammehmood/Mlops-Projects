import logging

from zenml import Model, pipeline

from steps.data_ingestion_step import data_ingestion_step
from steps.data_splitter_step import data_splitter_step
from steps.feature_engineering_step import feature_engineering_step
from steps.model_building_step import model_building_step
from steps.model_evaluator_step import model_evaluator_step


@pipeline(
    model=Model(name="salary_predictor"),
)
def ml_pipeline():
    """Define a corrected end-to-end machine learning pipeline."""

    # 1. Data Ingestion Step
    raw_data = data_ingestion_step(file_path="data/archive.zip")

    X_train, X_test, y_train, y_test = data_splitter_step(
        df=raw_data, target_column="salary_in_usd"
    )

    # 3. Feature Engineering Step (Fit on Train, Transform Train & Test)
    X_train_transformed, X_test_transformed = feature_engineering_step(
        X_train, X_test, y_train, y_test
    )

    # 4. Model Building Step
    model = model_building_step(X_train=X_train_transformed, y_train=y_train)

    # 5. Model Evaluation Step
    model_evaluator_step(trained_model=model, X_test=X_test_transformed, y_test=y_test)

    return model


if __name__ == "__main__":
    run = ml_pipeline()
    logging.info("Pipeline run completed successfully!")
