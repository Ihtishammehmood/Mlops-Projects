import pandas as pd
from zenml import step


@step
def dynamic_importer() -> str:
    """Dynamically imports data for testing out the model."""
    # Sample data that matches the training schema for salary prediction
    # data = {
    #     "work_year": [2023, 2022],
    #     "experience_level": ["SE", "MI"],
    #     "employment_type": ["FT", "CT"],
    #     "job_title": ["Data Scientist", "Data Engineer"],
    #     "employee_residence": ["US", "GB"],
    #     "remote_ratio": [100, 50],
    #     "company_location": ["US", "GB"],
    #     "company_size": ["L", "M"],
    # }

    df = pd.read_csv("extracted_data/salaries.csv").sample(10)

    # Convert the DataFrame to a JSON string
    json_data = df.to_json(orient="split")
    return json_data
