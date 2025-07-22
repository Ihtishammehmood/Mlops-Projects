# MLflow Experiment Tracking

![image](img/mflow.png)

# [See Full Youtube Video](https://www.youtube.com/watch?v=Y_XTjAyd_1U&t=72s)

## Introduction

This project demonstrates how to use MLflow for experiment tracking in machine learning workflows. MLflow is an open-source platform that helps manage the machine learning lifecycle, including experimentation, reproducibility, deployment, and a central model registry.

## Overview

The `MLflow-Experiment_Tracking` folder contains code and notebooks for tracking experiments using MLflow. It provides examples for logging parameters, metrics, and models, making it easier to compare different runs and experiments.

## Features

- **MLflow Integration:** Track parameters, metrics, and artifacts from your ML experiments.
- **Reproducibility:** Log each run for easy comparison and reproducibility.
- **Model Registry:** Option to register models for deployment and versioning.
- **Easy Setup:** Scripts and notebooks to get started with MLflow locally or on a remote server.

## Folder Structure

```
MLflow-Experiment_Tracking/
├── <scripts and notebooks>
├── requirements.txt
└── README.md
```

## Getting Started

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Ihtishammehmood/Mlops-Projects.git
   cd Mlops-Projects/MLflow-Experiment_Tracking
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run MLflow Tracking Example:**

   - Open the Jupyter notebook or execute the Python script to start experimenting.
   - MLflow UI can be started using:

     ```bash
     mlflow ui
     ```

   - Access the MLflow UI at [http://localhost:5000](http://localhost:5000).

## Example Usage

```python
import mlflow

with mlflow.start_run():
    mlflow.log_param("param1", 5)
    mlflow.log_metric("accuracy", 0.95)
    # Save your model
    mlflow.sklearn.log_model(model, "model")
```

## Requirements

- Python 3.7+
- MLflow
- (Other dependencies listed in `requirements.txt`)

## Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow Quickstart](https://mlflow.org/docs/latest/quickstart.html)

## License

This project is licensed under the MIT License.