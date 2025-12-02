# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI(title="Telco Customer Churn Prediction API")

# ----------------------------
# Load saved artifacts
# ----------------------------
MODEL_PATH = "final_churn_model.pkl"
PREPROCESSOR_PATH = "preprocessor.pkl"

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

# Try to load feature names to guide input expectations (optional)
try:
    import json
    with open("feature_names.json", "r") as f:
        feature_names = json.load(f)
except Exception:
    feature_names = None

# ----------------------------
# Pydantic input: accept a dict-like payload
# We don't hardcode every field to make it flexible; we accept JSON mapping of feature->value.
# ----------------------------
class CustomerData(BaseModel):
    data: dict  # {"gender": "Female", "SeniorCitizen": 0, ...}

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Telco Churn Prediction API is running."}

@app.post("/predict")
def predict(request: CustomerData):
    # Convert dict to single-row DataFrame
    try:
        row = request.data
        if not isinstance(row, dict):
            raise ValueError("data must be a dict mapping features to values")

        X = pd.DataFrame([row])

        # Ensure numeric columns are numeric (TotalCharges may be string)
        if "TotalCharges" in X.columns:
            X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")
        # If tenure or MonthlyCharges present ensure numeric:
        for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors="coerce")

        # Align columns: preprocessor expects a certain set of columns (fitting dataset's columns)
        # If any expected columns are missing, add them with default NaN (OneHotEncoder will handle unknowns)
        # To get expected input column order, we can infer from preprocessor input feature names.
        # But ColumnTransformer doesn't provide input column names directly; safer to add any missing columns found in training X columns.
        # We'll attempt to use the preprocessor's transformers to infer required columns:
        # Fallback: if preprocessor was fit on DataFrame columns, we assume unknown columns will be ignored and missing values allowed.

        # Transform
        X_trans = preprocessor.transform(X)
        proba = model.predict_proba(X_trans)[:, 1][0]
        pred = int(model.predict(X_trans)[0])

        return {
            "prediction": "Yes" if pred == 1 else "No",
            "probability_churn": float(proba)
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))