import json

import numpy as np
import pandas as pd
from zenml import step
from zenml.integrations.mlflow.services import MLFlowDeploymentService


@step(enable_cache=False)
def predictor(
    service: MLFlowDeploymentService,
    input_data: pd.DataFrame,
) -> np.ndarray:
    """Run an inference request against a prediction service.

    Args:
        service (MLFlowDeploymentService): The deployed MLFlow service for prediction.
        input_data (pd.DataFrame): The input data as a pd.DataFrame.

    Returns:
        np.ndarray: The model's prediction.
    """

    # Start the service (should be a NOP if already started)
    service.start(timeout=10)

    # # Load the input data from JSON string
    # data = json.loads(input_data)

    # # Extract the actual data and expected columns
    # data.pop("columns", None)  # Remove 'columns' if it's present
    # data.pop("index", None)  # Remove 'index' if it's present

    # # Define the columns the model expects
    # expected_columns = [
    #     'work_year',
    #     'experience_level',
    #     'employment_type',
    #     'job_title',
    #     'salary_in_usd',
    #     'employee_residence',
    #     'work_status',
    #     'company_location',
    #     'company_size']

    # # Convert the data into a DataFrame with the correct columns
    # df = pd.DataFrame(data["data"], names=expected_columns)
    df = input_data

    # Convert DataFrame to JSON list for prediction
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    data_array = np.array(json_list)

    # Run the prediction
    prediction = service.predict(data_array)

    return prediction