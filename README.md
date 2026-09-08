
# Telco Customer Churn Prediction

A Machine Learning project that predicts whether a telecom customer is likely to churn.

## Project Overview

This project uses customer information to predict customer churn using a trained XGBoost machine learning model.

The project includes:

- Data Cleaning
- Exploratory Data Analysis
- Feature Engineering
- Data Preprocessing
- Machine Learning Model Training
- Cross Validation
- Hyperparameter Tuning
- Model Evaluation
- FastAPI Deployment

## Project Structure

```text
telco-churn-ml/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── schemas.py
│   └── model.py
│
├── models/
│   └── telco_churn_xgb_pipeline.pkl
│
├── requirements.txt
├── .gitignore
├── README.md
└── Dockerfile
