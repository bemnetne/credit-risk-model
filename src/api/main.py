import mlflow
import mlflow.sklearn

import pandas as pd

from fastapi import FastAPI

from src.api.pydantic_models import (
    PredictionRequest,
    PredictionResponse
)
mlflow.set_tracking_uri(
    "sqlite:///notebooks/mlflow.db"
)
print("Tracking URI:", mlflow.get_tracking_uri())
app = FastAPI(
    title="Customer Credit Risk API",

    description="""
    Predicts the probability that a customer
    belongs to the high-risk segment based
    on historical transaction behavior.

    The API also generates a credit score and
    loan recommendation for Buy Now Pay Later
    decision support.
    """
)

MODEL_NAME = "CreditRiskModel"

model = mlflow.sklearn.load_model(
    f"models:/{MODEL_NAME}/latest"
)
def probability_to_score(probability):

    if probability < 0.15:
        return 850

    elif probability < 0.20:
        return 750

    elif probability < 0.25:
        return 650

    elif probability < 0.30:
        return 550

    else:
        return 450
def recommend_loan(credit_score):

    if credit_score >= 800:
        return {"limit": 100000, "duration": 120}

    elif credit_score >= 750:
        return {"limit": 50000, "duration": 90}

    elif credit_score >= 700:
        return {"limit": 25000, "duration": 60}

    elif credit_score >= 600:
        return {"limit": 10000, "duration": 45}

    else:
        return {"limit": 5000, "duration": 30}
        
@app.get("/")
def health_check():

    return {
        "status": "healthy",
        "model": MODEL_NAME
    }
@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(
    request: PredictionRequest
):
    data = pd.DataFrame([
        request.model_dump()
    ])

    probability = (
        model.predict_proba(data)[0][1]
    )

    prediction = (
        model.predict(data)[0]
    )

    credit_score = (
    probability_to_score(
        probability
    )
    )

    loan = recommend_loan(
        credit_score
    )

    return PredictionResponse(
        risk_probability=float(
        probability
    ),

    risk_prediction=int(
        prediction
    ),

    credit_score=credit_score,

    recommended_limit=loan[
        "limit"
    ],

    recommended_duration_days=loan[
        "duration"
    ]
)