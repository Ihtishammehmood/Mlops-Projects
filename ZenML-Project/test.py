import pandas as pd
import json

df = pd.read_csv('extracted_data/salaries.csv')

def unique_values(data):
    
    result = {}
    columns_names = ['work_year', 'experience_level', 'employment_type', 'job_title',
                     'employee_residence', 'work_status', 'company_location', 'company_size']
    for col in data.columns:
        if col in columns_names:
            result[col] = data[col].unique().tolist()  # Convert NumPy array to list
    return result

if __name__ == "__main__":
    unique_vals = unique_values(df)
    print(unique_vals)
    with open("unique_values.json", "w") as outfile:
        json.dump(unique_vals, outfile, indent=4)