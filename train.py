import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("career_prediction_dataset.csv")

# Define target variable and features
TARGET_COLUMN = "Suggested Career Path"
FEATURE_COLUMNS = [col for col in df.columns if col != TARGET_COLUMN]

# Separate features (X) and target variable (y)
X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMN]

# Encode target variable (Suggested Career Path)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # Convert career names to numbers

# Save label encoder for decoding predictions later
with open("label_encoder.pkl", "wb") as le_file:
    pickle.dump(label_encoder, le_file)

# Encode categorical columns
categorical_columns = X.select_dtypes(include=['object']).columns
label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])  # Convert categorical to numerical
    label_encoders[col] = le

# Save encoders for inference
with open("label_encoders.pkl", "wb") as le_dict_file:
    pickle.dump(label_encoders, le_dict_file)

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler for inference
with open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save trained model
with open("career_prediction_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

print("Model training completed and saved!")
