import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

# ============================
# 1. Load cleaned dataset
# ============================
df = pd.read_csv("Telco_Customer_Churn_CLEANED.csv")

df["Churn_flag"] = df["Churn"].map({"Yes": 1, "No": 0})

X = df.drop(columns=["Churn", "Churn_flag"])
y = df["Churn_flag"]

# ============================
# 2. Preprocessing
# ============================
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), num_cols)
    ],
    remainder="drop"
)

# ============================
# 3. Train-test split
# ============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Fit preprocessor
preprocessor.fit(X_train)

# Transform data
X_train_t = preprocessor.transform(X_train)
X_test_t = preprocessor.transform(X_test)

# ============================
# 4. Train Model
# ============================
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train_t, y_train)

# ============================
# 5. Save artifacts
# ============================
joblib.dump(preprocessor, "preprocessor.pkl")
joblib.dump(model, "final_churn_model.pkl")

print("✓ preprocessor.pkl created")
print("✓ final_churn_model.pkl created")