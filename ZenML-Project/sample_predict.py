import json
import pandas as pd
import requests
# Import the data_preprocessor function
from steps.data_preprocessor import data_preprocessor

# URL of the MLflow prediction server
url = "http://localhost:8000/invocations"

# Sample input data for prediction
input_data = {
            "work_year": [2025],
            "experience_level": ["EN"],
            "employment_type": ["FT"],
            "job_title": ["Applied Scientist"],
            "employee_residence": ["US"],
            "remote_ratio": [0],
            "company_location": ["US"],
            "company_size": ["M"],
        }

# Convert to dataframe
# df = pd.DataFrame(input_data)

# Preprocess the data
# The data_preprocessor ZenML step expects a pandas DataFrame and returns a
# preprocessed DataFrame.
try:
    # Parse the preprocessed JSON string back into a Python dictionary
    preprocessed_data_dict = json.loads(input_data)
    
    # Wrap it in the format MLflow expects
    payload = {"dataframe_split": preprocessed_data_dict}
    
    # Use the 'json' parameter to send the payload
    response = requests.post(url, json=payload)

    # Check the response status code
    if response.status_code == 200:
        # If successful, print the prediction result
        prediction = response.json()
        print("Prediction:", prediction)
    else:
        # If there was an error, print the status code and the response
        print(f"Error: {response.status_code}")
        print(response.text)

except Exception as e:
    print(f"An error occurred: {e}")