import streamlit as st
import requests
import json

# FastAPI endpoint
url = 'http://127.0.0.1:8000/diabetes_prediction'

st.title("🩺 Diabetes Prediction App")
st.markdown("Fill the details below to check if the person is diabetic or not.")

# Input fields
pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
glucose = st.number_input("Glucose", min_value=0, step=1)
blood_pressure = st.number_input("Blood Pressure", min_value=0, step=1)
skin_thickness = st.number_input("Skin Thickness", min_value=0, step=1)
insulin = st.number_input("Insulin", min_value=0, step=1)
bmi = st.number_input("BMI", min_value=0.0, step=0.1, format="%.1f")
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, step=0.01)
age = st.number_input("Age", min_value=0, step=1)

# Predict button
if st.button("Predict"):
    input_data = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    }

    try:
        response = requests.post(url, data=json.dumps(input_data))
        st.success(f"Prediction Result: {response.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to API: {e}")
