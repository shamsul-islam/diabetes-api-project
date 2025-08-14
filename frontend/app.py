
import streamlit as st
import requests
import json

# Set the title of the app
st.title("Diabetes Prediction")

# Create the form for user input
with st.form("prediction_form"):
    st.header("Enter Patient Data")

    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=3)
    glucose = st.number_input("Glucose", min_value=0, max_value=200, value=145)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
    insulin = st.number_input("Insulin", min_value=0, max_value=900, value=85)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=33.6)
    diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.35)
    age = st.number_input("Age", min_value=0, max_value=120, value=29)

    submitted = st.form_submit_button("Predict")

# When the user clicks the predict button
if submitted:
    # Create a dictionary from the user input
    patient_data = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": diabetes_pedigree_function,
        "Age": age
    }

    # Send a POST request to the FastAPI backend
    # IMPORTANT: Replace with your deployed Render URL
    api_url = "http://127.0.0.1:8000/predict"
    try:
        response = requests.post(api_url, json=patient_data)
        response.raise_for_status()  # Raise an exception for bad status codes
        prediction = response.json()

        # Display the prediction result
        st.header("Prediction Result")
        st.write(f"**Prediction:** {prediction['result']}")
        st.write(f"**Confidence:** {prediction['confidence']:.2f}")

    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to the API: {e}")

