
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.schemas import Customer
from app.model import predict_churn


app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="AI-powered telecom customer churn prediction using XGBoost.",
    version="1.0.0"
)


# Serve CSS and JavaScript files
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def home():
    return FileResponse("static/index.html")


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
