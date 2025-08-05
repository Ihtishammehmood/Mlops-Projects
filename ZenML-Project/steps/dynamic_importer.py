import pandas as pd
from zenml import step

from steps.data_preprocessor import data_preprocessor


@step
def dynamic_importer(path: str) -> str:
    """Dynamically imports data for testing out the model."""

    df = pd.read_csv(path).sample(10)
    expected_columns = [
        "work_year",
        "experience_level",
        "employment_type",
        "job_title",
        "employee_residence",
        "remote_ratio",
        "company_location",
        "company_size",
    ]

    df = df[expected_columns]
    json_data = df.to_json(orient="split")
    transformed_data = data_preprocessor(json_data=json_data)
    return transformed_data
