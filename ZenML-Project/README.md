# ZenML Machine Learning Pipeline Project

This project implements a machine learning pipeline using ZenML for orchestration, MLflow for experiment tracking and model deployment.

## Project Overview

The project is structured as a complete ML pipeline that includes data ingestion, preprocessing, feature engineering, model training, evaluation, and deployment steps.

## Data source

ML model is trained on `Data science and AI salary in 2025` dataset from Kaggle. You can access the dataset [here](https://www.kaggle.com/datasets/adilshamim8/salaries-for-data-science-jobs)


## Prerequisites

- Python 3.13
- ZenML
- MLflow
- Streamlit
- Other dependencies listed in `requirements.txt`

## Project Structure

```plaintext
├── artifacts/             # Saved model artifacts and scalers
├── data/                  # Raw data files
├── extracted_data/        # Processed data files
├── pipelines/             # Pipeline definitions
├── src/                   # Source code for ML operations
└── steps/                 # Individual pipeline steps
```

## Setup and Installation

1. Clone the repository `https://github.com/Ihtishammehmood/Mlops-Projects.git` and `cd ZenML-Project`

2. Install uv package manager `pip install uv`
3. create virtual environment `uv venv` and activate it `.venv\Scripts\activate`
4. Install dependencies `uv sync`

5. Set up ZenML:

   ```bash
   # Initialize  ZenML repository
   zenml init
   
   # to access zenml dasboard in windows
   zenml login --local --blocking

6. Install and configure integrations:

   ```bash
   # Install required integrations
   zenml integration install mlflow -y --uv

   # Register model deployer
   zenml model-deployer register mlflow --flavor=mlflow

   # Register experiment tracker
   zenml experiment-tracker register mlflow_tracker --flavor=mlflow

   # Set up the stack
   zenml stack register local-mlflow-stack -a default -o default -d mlflow -e mlflow_tracker --set
   
   ```

## Running the Pipeline

1. Training Pipeline:

   ```bash
   uv run run_pipeline.py
   ```

2. Deployment Pipeline:
 - daemon functionality is not natively supported on Windows so you need to run the deployment inside `docker` or `wsl`
 - build docker `docker built -t image_name .`
 - Get inside the docker `docker run -it --entrypoint /bin/bash image_name`

   ```bash
   uv run run_deployment.py
   ```

- Make Predictions:

   ```bash
   uv run sample_predict.py # run this inside the docker container
   ```
3. To inference on the streamlit app:

   ```bash
   uv run streamlit run client.py
   ```
## Pipeline Components

- Data Ingestion: Loads and processes raw data
- Feature Engineering: Creates and transforms features
- Model Training: Trains the ML model
- Model Evaluation: Evaluates model performance
- Model Deployment: Deploys model using mlfow

## MLflow Integration

The project uses MLflow for:

- Experiment tracking
- Model versioning
- Metrics logging
- Model registry

## Contributing

Feel free to open issues or submit pull requests for improvements.

## License

This project is licensed under the MIT License.
