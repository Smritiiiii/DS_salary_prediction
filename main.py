import streamlit as st
import joblib
import pandas as pd

model = joblib.load('age_salary_prediction_model.pkl')

trained_feature = model.feature_names_in_[1:]  

st.title('Salary Prediction App')

age = st.number_input('Enter your Age', min_value=18, max_value=85)
job_title = st.selectbox('Select your Job Title', trained_feature)


if job_title and age:
    # age = float(age_input)

    job_data = pd.DataFrame([[0] * len(trained_feature)], columns=trained_feature)

    job_data[job_title] = 1
    
    job_data['Age'] = age

    job_data = job_data[['Age'] + list(trained_feature)]

    if st.button('Predict Salary'):
       
        salary_pred = model.predict(job_data)
        st.success(f'Predicted Salary: {salary_pred[0]}')
    else:
        st.error("Wrong input")
