# Customer Churn Prediction API
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg?style=flat-square)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688.svg?style=flat-square)](https://fastapi.tiangolo.com/)
[![Model](https://img.shields.io/badge/ML-LogisticRegression-orange.svg?style=flat-square)]()
[![Status](https://img.shields.io/badge/status-active-brightgreen.svg?style=flat-square)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg?style=flat-square)]()

---

## Table of Contents

- [About](#about)  
- [Features](#features)  
- [Repository Structure](#repository-structure)  
- [Prerequisites](#prerequisites)  
- [Installation & Setup](#installation--setup)  
- [Running the API](#running-the-api)  
- [API Usage Example](#api-usage-example)  
- [Model Details](#model-details)  
- [Retraining the Model](#retraining-the-model)  
- [Deployment](#deployment)  
- [Roadmap / Future Work](#roadmap--future-work)  
- [Contributing](#contributing)  
- [License](#license)  

---

## About

The Customer Churn Prediction API is a machine-learning powered backend service for predicting whether a customer is likely to churn (leave) based on their account data and usage metrics. The service includes data preprocessing, a trained ML model, and a RESTful endpoint to make inference easy and quick — ideal for integrating into customer-management or CRM systems.

---

## Features

- Predicts probability of customer churn using historical customer data  
- Provides a ready-to-use REST API for real-time predictions  
- Includes preprocessing pipeline (encoding, scaling)  
- Model persistence with `.pkl` files for reuse and portability  
- Easy retraining option with `train_model.py`  
- Lightweight and easy to deploy  

---

## Repository Structure

**Customer Churn Prediction API**

A simple ML-powered web service that predicts customer churn for a telecom business — built with FastAPI and a trained scikit-learn model.

**Project Overview**

This project uses historical customer data to train a classification model that predicts whether a customer is likely to churn. The trained model (plus preprocessor) is exposed via an HTTP API so new customer data can be submitted and churn-risk predictions returned in real time.

 **Repository Structure:**
 ```
 
├── Telco_Customer_Churn_cleaned.csv     # Dataset used for model training
├── train_model.py                        # Script for training model & preprocessing pipeline
├── final_churn_model.pkl                 # Saved trained machine learning model
├── preprocessor.pkl                      # Saved preprocessing transformer
├── app.py                                # FastAPI app for model inference
├── requirements.txt                       # Project dependencies
├── Procfile                               # Deployment config 
└── README.md                              # Documentation

```

**Installation & Setup:**

Step 1 — Clone the Repository

git clone https://github.com/day6ix/customer-churn-prediction-API.git
cd customer-churn-prediction-API

Step 2 — Install Dependencies
```
pip install -r requirements.txt

```

---

## Prerequisites

- Python 3.10 or higher  
- Basic Python packages (as listed in `requirements.txt`)  
- (Optional) Virtual environment tool (venv, conda, etc.)  

---

## Installation & Setup

```bash
# 1. Clone repo
git clone https://github.com/day6ix/customer-churn-prediction-API.git
cd customer-churn-prediction-API

# 2. (Optional) Create and activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
# source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

```

**Running the API**

**Start the local API server:**

```
uvicorn app:app --reload

```
**The API will run at:**

```
http://127.0.0.1:8000

```

**Docs UI available at:**
```
http://127.0.0.1:8000/docs

```

**Example Prediction Request:**

POST /predict
```

{
  "tenure": 12,
  
  "MonthlyCharges": 70.35,
  
  "Contract": "Month-to-month",
  
  "OnlineSecurity": "No",
  
  "TechSupport": "No",
  
  "InternetService": "Fiber optic",
  
  "gender": "Female",
  
  "SeniorCitizen": 0,
  
  "Partner": "Yes",
  
  "Dependents": "No"
  
}

```
**Example Response:**
```
{

  "churn_probability": 0.86,
  
  "churn_prediction": true
  
}

```
**Model Details**

	•	Framework: scikit-learn
	•	Preprocessing: encoding of categorical variables, scaling of numeric features
	•	Serialization: model and preprocessor saved with pickle as .pkl files
	•	Inference: Preprocessor transforms raw features, model predicts churn probability & class
  

**You can retrain the model using:**

If you want to retrain using updated or different data:

```
python train_model.py

```
This will regenerate preprocessor.pkl and final_churn_model.pkl based on Telco_Customer_Churn_cleaned.csv (or new data if updated).
Ensure data schema matches requirements of training pipeline.

**Deployment**

**This API can be deployed on:**

	•	Render
  
	•	Railway
  
	•	Heroku
  
	•	Azure
  
	•	AWS EC2 / Lambda
  
	•	Google Cloud Run
  
	•	Docker container

(Procfile already included for Heroku-style deployment)

**Roadmap / Future Work:**

	•	Add input validation & error handling for API
	
	•	Add unit tests for prediction and preprocessing logic
	
	•	Add model evaluation metrics & logging (e.g. accuracy, ROC-AUC)
	
	•	Support batch predictions (multiple customers at once)
	
	•	Add versioning & change log for model updates
	
	•	(Optional) Add authentication/authorization for API requests

 **Contributing**
 
	1.	Fork the repository
  
	2.	Create a feature branch
  
	3.	Commit your changes
  
	4.	Submit a pull request

Open issues if you find bugs or have improvement ideas!

**License**

This project is licensed under the MIT License.


**Contact**

For suggestions or questions, feel free to reach out via GitHub issues section.
