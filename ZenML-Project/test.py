import pandas as pd
from steps.dynamic_importer import dynamic_importer
from steps.model_loader import model_loader


model = model_loader('salary_predictor')




if __name__ == "__main__":
    print(model)