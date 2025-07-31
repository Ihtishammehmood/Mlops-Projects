import pandas as pd
from zenml import step
from steps.data_preprocessor import data_preprocessor
import json

@step
def dynamic_importer(path:str) -> str:
    """Dynamically imports data for testing out the model."""

    df = pd.read_csv(path).sample(10)
    expected_columns = [
        'work_year',
        'experience_level',
        'employment_type',
        'job_title',
        'employee_residence',
        'remote_ratio',
        'company_location',
        'company_size']
    
    # add expected columns only to dataframe
    df = df[expected_columns]

    # Convert the DataFrame to a JSON string
    json_data = df.to_json(orient="split")

    # apply all the expected transformations to the dataframe
    transformed_data  = data_preprocessor(json_data=json_data)

    return transformed_data