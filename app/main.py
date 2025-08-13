
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

app = FastAPI()

# Load the model and test data
model = joblib.load('model/diabetes_model.pkl')
df = pd.read_csv('data/diabetes.csv')
X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class PatientData(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(data: PatientData):
    input_data = np.array([[data.Pregnancies, data.Glucose, data.BloodPressure, data.SkinThickness, data.Insulin, data.BMI, data.DiabetesPedigreeFunction, data.Age]])
    prediction = model.predict(input_data)[0]
    confidence = model.predict_proba(input_data).max()

    result = "Diabetic" if prediction == 1 else "Not Diabetic"

    return {
        "prediction": int(prediction),
        "result": result,
        "confidence": float(confidence)
    }

@app.get("/metrics")
def metrics():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
