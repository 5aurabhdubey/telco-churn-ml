
import os
import joblib
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "telco_churn_xgb_pipeline.pkl"
)


model = joblib.load(MODEL_PATH)


def predict_churn(customer_data: dict):

    # Convert dictionary into DataFrame
    customer_df = pd.DataFrame([customer_data])

    # Get probability of churn
    probability = model.predict_proba(customer_df)[0][1]

    # Default threshold = 0.5
    prediction = int(probability >= 0.5)

    result = "Yes" if prediction == 1 else "No"

    return {
        "churn_prediction": result,
        "churn_probability": round(float(probability), 4)
    }
