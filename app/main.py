
from fastapi import FastAPI

from app.schemas import Customer
from app.model import predict_churn


app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="API for predicting telecom customer churn using XGBoost.",
    version="1.0.0"
)


@app.get("/")
def home():
    return {
        "message": "Telco Churn Prediction API is running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(customer: Customer):

    customer_data = customer.model_dump()

    result = predict_churn(customer_data)

    return result
