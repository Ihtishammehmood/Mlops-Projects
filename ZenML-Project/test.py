import requests
import pandas as pd
from steps.data_preprocessor import data_preprocessor

input_data = {
    "dataframe_records": [
        {
            "work_year": 2025,
            "experience_level": "EN",
            "employment_type": "FT",
            "job_title": "Applied Scientist",
            "employee_residence": "US",
            "remote_ratio": 0,
            "company_location": "US",
            "company_size": "M",
        }
    ]
}

df = pd.DataFrame(input_data["dataframe_records"])
preprocessed = data_preprocessor(df)  # returns DataFrame with proper columns
payload = {
    "inputs": preprocessed.to_dict(orient="records")[0]
}


if __name__ == "__main__":
    print(payload)