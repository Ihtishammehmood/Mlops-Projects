import pandas as pd
from zenml import step


@step
def dynamic_importer(path:str) -> str:
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

    df = pd.read_csv(path).sample(10)
    expected_columns = [
        'work_year',
        'experience_level',
        'employment_type',
        'job_title',
        'salary_in_usd',
        'employee_residence',
        'remote_ratio',
        'company_location',
        'company_size']
    
    # add expected columns only to dataframe
    df = df[expected_columns]

    # Convert the DataFrame to a JSON string
    json_data = df.to_json(orient="split")
    return json_data
