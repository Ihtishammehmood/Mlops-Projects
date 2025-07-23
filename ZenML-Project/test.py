import pandas as pd
# import tabulate as tb

expected_columns = [
            'work_year',
            'experience_level',
            'employment_type',
            'job_title',
            'salary_in_usd',
            'employee_residence',
            'work_status',
            'company_location',
            'company_size']

df = pd.read_csv('extracted_data/salaries.csv',names = expected_columns).sample(10)







if __name__ == "__main__":
    print(df)
        