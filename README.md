# Diabetes Prediction API & Frontend

This project implements a machine learning model to predict diabetes, exposed via a FastAPI backend, containerized with Docker, and consumed by a Streamlit frontend.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Setup and Installation](#setup-and-installation)
- [How to Run Locally](#how-to-run-locally)
- [API Endpoints](#api-endpoints)
- [Deployment](#deployment)
- [Project Structure](#project-structure)
- [License](#license)

## Project Overview

This project aims to predict diabetes based on patient diagnostic measurements. It involves:
1.  Training a classification model using the Pima Indians Diabetes Dataset.
2.  Developing a FastAPI application to serve predictions.
3.  Containerizing the FastAPI application using Docker.
4.  Building a simple Streamlit frontend for user interaction.

## Features

-   **Machine Learning Model:** Trains and evaluates multiple classification models (Logistic Regression, Random Forest) to predict diabetes.
-   **FastAPI Backend:**
    -   Provides a `/health` endpoint to check API status.
    -   Offers an asynchronous `/predict` endpoint for real-time diabetes predictions.
    -   Includes a `/metrics` endpoint to display model evaluation metrics.
-   **Dockerization:** The FastAPI application is containerized for easy deployment and portability.
-   **Streamlit Frontend:** A user-friendly web interface to input patient data and display prediction results (Diabetic/Not Diabetic) along with confidence scores.

## Technologies Used

-   **Python 3.9+**
-   **FastAPI:** For building the web API.
-   **Uvicorn:** ASGI server for FastAPI.
-   **Scikit-learn:** For machine learning model training and evaluation.
-   **Pandas:** For data manipulation.
-   **NumPy:** For numerical operations.
-   **Joblib:** For saving and loading the trained model.
-   **Streamlit:** For building the interactive web frontend.
-   **Requests:** For making HTTP requests from the frontend to the backend.
-   **Docker:** For containerization.

## Setup and Installation

To set up the project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/diabetes-prediction-api.git
    cd diabetes-prediction-api
    ```
    *(Replace `YOUR_USERNAME` with your GitHub username and `diabetes-prediction-api` with your repository name)*

2.  **Create and activate a virtual environment:**
    ```bash
    # For Windows
    python -m venv .venv
    .venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install backend dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install frontend dependencies:**
    ```bash
    pip install -r frontend/requirements.txt
    ```

## How to Run Locally

### 1. Train the Model

First, train the machine learning model. This will save `diabetes_model.pkl` in the `model/` directory.

```bash
python model/train_model.py
```

### 2. Run the FastAPI Backend

Start the FastAPI server. Ensure your virtual environment is activated.

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```
The API will be accessible at `http://127.0.0.1:8000`.

### 3. Run the Streamlit Frontend

Open a **new terminal** (keep the backend running in the first one) and activate your virtual environment. Then, run the Streamlit app:

```bash
streamlit run frontend/app.py
```
The frontend will typically open in your browser at `http://localhost:8501`.

## API Endpoints

The FastAPI backend exposes the following endpoints:

-   **GET `/health`**
    -   **Description:** Checks the health status of the API.
    -   **Response:** `{"status": "ok"}`

-   **POST `/predict`**
    -   **Description:** Predicts whether a patient has diabetes based on input features.
    -   **Request Body (JSON):**
        ```json
        {
          "Pregnancies": 3,
          "Glucose": 145,
          "BloodPressure": 70,
          "SkinThickness": 20,
          "Insulin": 85,
          "BMI": 33.6,
          "DiabetesPedigreeFunction": 0.35,
          "Age": 29
        }
        ```
    -   **Response (JSON):**
        ```json
        {
          "prediction": 0,
          "result": "Not Diabetic",
          "confidence": 0.87
        }
        ```

-   **GET `/metrics`**
    -   **Description:** Returns the evaluation metrics (accuracy, precision, recall, F1 score) of the trained model on the test set.
    -   **Response (JSON):**
        ```json
        {
          "accuracy": 0.7468,
          "precision": 0.6379,
          "recall": 0.6727,
          "f1_score": 0.6549
        }
        ```

## Deployment

This project is designed for deployment using Docker and platforms like Render for the backend, and Streamlit Cloud for the frontend.

-   **Live Render API:** [YOUR_RENDER_API_URL_HERE]
-   **Live Streamlit Frontend:** [YOUR_STREAMLIT_APP_URL_HERE]

*(Remember to replace the placeholder URLs above with your actual deployed links.)*

## Project Structure

```
.
├── .gitignore
├── Dockerfile
├── Module12_Project.txt
├── requirements.txt
├── app/
│   └── main.py
├── data/
│   └── diabetes.csv
├── model/
│   ├── diabetes_model.pkl
│   └── train_model.py
└── frontend/
    ├── app.py
    └── requirements.txt
```

## License

This project is open-source and available under the MIT License.
