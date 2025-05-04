from fastapi import FastAPI
from pydantic import BaseModel
import pickle

app = FastAPI()

class ModelInput(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

# Load model
with open("model/diabetes_model.pkl", "rb") as f:
    diabetes_model = pickle.load(f)

@app.post("/diabetes_prediction")
def predict_diabetes(input_data: ModelInput):
    input_list = [
        input_data.Pregnancies, input_data.Glucose, input_data.BloodPressure,
        input_data.SkinThickness, input_data.Insulin, input_data.BMI,
        input_data.DiabetesPedigreeFunction, input_data.Age
    ]

    prediction = diabetes_model.predict([input_list])
    return {
        "prediction": "The person is Diabetic" if prediction[0] == 1 else "The person is not Diabetic"
    }
