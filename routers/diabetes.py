from fastapi import APIRouter
from models import load_model
from schemas.diabetes import DiabetesInput
import numpy as np

router = APIRouter()

model = load_model("models/diabetes_model.pkl")

@router.post("/diabetes_prediction")
def predict_diabetes(data: DiabetesInput):
    input_arr = np.array([[
        data.Pregnancies, data.Glucose, data.BloodPressure,
        data.SkinThickness, data.Insulin, data.BMI,
        data.DiabetesPedigreeFunction, data.Age
    ]])
    prediction = model.predict(input_arr)
    return {"result": "Diabetic" if prediction[0] == 1 else "Not Diabetic"}
