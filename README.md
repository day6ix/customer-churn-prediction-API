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

	•	Built using scikit-learn
  
	•	Uses a preprocessing pipeline (encoding + scaling)
  
	•	Supports reproducible training via train_model.py
  

**You can retrain the model using:**
```
python train_model.py

```

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

 **Contributing**
 
	1.	Fork the repository
  
	2.	Create a feature branch
  
	3.	Commit your changes
  
	4.	Submit a pull request

Open issues if you find bugs or have improvement ideas!

**Contact**

For suggestions or questions, feel free to reach out via GitHub issues section.
