import streamlit as st
import pandas as pd
import numpy as np
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Title
st.title("Student Performance Prediction App")
st.write("Enter the required information below to predict performance.")

# Sidebar for input fields
st.sidebar.header("Student Information")

gender = st.sidebar.selectbox("Gender", ["male", "female"])
race_ethnicity = st.sidebar.selectbox("Race/Ethnicity", [
    "group A", "group B", "group C", "group D", "group E"
])
parental_level_of_education = st.sidebar.selectbox("Parental Level of Education", [
    "some high school", "high school", "some college", "associate's degree",
    "bachelor's degree", "master's degree"
])
lunch = st.sidebar.selectbox("Lunch", ["standard", "free/reduced"])
test_preparation_course = st.sidebar.selectbox("Test Preparation Course", [
    "none", "completed"
])
reading_score = st.sidebar.number_input("Reading Score", min_value=0, max_value=100, value=50)
writing_score = st.sidebar.number_input("Writing Score", min_value=0, max_value=100, value=50)

# Prediction button
if st.button("Predict"):
    try:
        # Create CustomData object
        data = CustomData(
            gender=gender,
            race_ethnicity=race_ethnicity,
            parental_level_of_education=parental_level_of_education,
            lunch=lunch,
            test_preparation_course=test_preparation_course,
            reading_score=reading_score,
            writing_score=writing_score
        )

        pred_df = data.get_data_as_data_frame()

        st.write("### Input Data")
        st.dataframe(pred_df)

        # Prediction pipeline
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        st.success(f"Predicted Score: {results[0]:.2f}")

    except Exception as e:
        st.error(f"Error: {e}")
