import json
import requests

url = "http://127.0.0.1:8000/diabetes_prediction"

input_data = {
    "Pregnancies": 6,
    "Glucose": 148,
    "BloodPressure": 72,
    "SkinThickness": 35,
    "Insulin": 0,
    "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.627,
    "Age": 50
}

response = requests.post(url, json=input_data)
print("Prediction:", response.json())
