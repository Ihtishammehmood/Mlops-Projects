# Work In Process....

- run zenml server

```
zenml login --local --blocking

```

- run zenml in docker

```
zenml login --local --docker
```

```bash
# Install integrations
zenml integration install bentoml mlflow -y --uv
zenml integration install mlflow -y --uv

# Register BentoML model deployer
zenml model-deployer register bentoml_deployer --flavor=bentoml

# Register MLflow experiment tracker
zenml experiment-tracker register mlflow_tracker --flavor=mlflow

# Register and set the stack (add other components as needed)
# zenml stack register my_stack -d bentoml_deployer -e mlflow_tracker -a default -o default

zenml stack register local-mlflow-stack -a default -o default -d mlflow -e mlflow_tracker --set

zenml stack set my_stack


```
