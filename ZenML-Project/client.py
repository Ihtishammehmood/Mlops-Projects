import streamlit as st
import pandas as pd
import json
import numpy as np
import os
import joblib
from steps.data_preprocessor import data_preprocessor

# Load the trained model
model = joblib.load('artifacts/model_trainer.pkl')

# Load unique values from JSON
with open('unique_values.json', 'r') as f:
    unique_values = json.load(f)

st.title('AI/ML Salary Prediction App 💰')
st.write('Enter the details below to predict the salary for a position')

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        work_year = st.selectbox(
            'Work Year',
            options=sorted(unique_values['work_year'], reverse=True),
            help='The year for which you want to predict the salary'
        )

        experience_level = st.selectbox(
            'Experience Level',
            options=unique_values['experience_level'],
            help='EN=Entry Level, MI=Mid Level, SE=Senior, EX=Executive'
        )

        employment_type = st.selectbox(
            'Employment Type',
            options=unique_values['employment_type'],
            help='FT=Full-Time, PT=Part-Time, CT=Contract, FL=Freelance'
        )

        job_title = st.selectbox(
            'Job Title',
            options=sorted(unique_values['job_title']),
            help='Select the job title that best matches the position'
        )

    with col2:
        employee_residence = st.selectbox(
            'Employee Residence',
            options=sorted(unique_values['employee_residence']),
            help='Country where the employee resides'
        )

        remote_ratio = st.select_slider(
            'Remote Work Ratio',
            options=[0, 50, 100],
            help='0=No remote work (onsite), 50=Partially remote, 100=Fully remote'
        )

        company_location = st.selectbox(
            'Company Location',
            options=sorted(unique_values['company_location']),
            help='Country where the company is located'
        )

        company_size = st.selectbox(
            'Company Size',
            options=unique_values['company_size'],
            help='S=Small, M=Medium, L=Large'
        )

    submitted = st.form_submit_button("Predict Salary")

if submitted:
    # Prepare input data
    input_data = {
        "dataframe_records": [
            {
                "work_year": work_year,
                "experience_level": experience_level,
                "employment_type": employment_type,
                "job_title": job_title,
                "employee_residence": employee_residence,
                "remote_ratio": remote_ratio,
                "company_location": company_location,
                "company_size": company_size,
            }
        ]
    }

    try:
        # Create DataFrame and preprocess
        df = pd.DataFrame(input_data["dataframe_records"])
        preprocessed = data_preprocessor(df)
        
        with st.spinner('Predicting salary...'):
            # Make prediction using the loaded model
            prediction = model.predict(preprocessed)
            prediction = np.exp(prediction)
            predicted_salary = round(float(prediction[0]), 2)
            st.success(f'Predicted Salary: ${predicted_salary:,.2f} USD')
            
            # Additional insights
            st.info("""
            **Factors that influenced this prediction:**
            - Experience Level & Job Title are typically the strongest predictors
            - Remote work ratio can impact salary expectations
            - Company size and location play significant roles
            """)
    except Exception as e:
        st.error(f'Error occurred: {str(e)}')

# Add some helpful information at the bottom
st.markdown("""
---
### About this Predictor
This salary prediction tool uses machine learning to estimate salaries based on various factors. 
The model was trained on historical salary data and considers multiple factors including:
- Work experience level
- Employment type
- Job title
- Geographic locations
- Company size
- Remote work arrangements

**Note**: Predictions are estimates and actual salaries may vary based on additional factors.
""")
