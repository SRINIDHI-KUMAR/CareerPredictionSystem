from flask import Flask, request, jsonify, render_template, redirect, url_for
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model and preprocessing tools
with open("career_prediction_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("label_encoder.pkl", "rb") as le_file:
    label_encoder = pickle.load(le_file)

with open("label_encoders.pkl", "rb") as le_dict_file:
    label_encoders = pickle.load(le_dict_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Load dataset to get feature names
df = pd.read_csv("career_prediction_dataset.csv")
TARGET_COLUMN = "Suggested Career Path"
FEATURE_COLUMNS = [col for col in df.columns if col != TARGET_COLUMN]

@app.route("/")
def root_redirect():
    return redirect(url_for("welcome"))  # Redirects to /welcome

@app.route("/home")
def home():
    return render_template("index.html")

@app.route("/welcome")
def welcome():
    return render_template("welcome.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form.to_dict()
        print("Received data:", data)

        df_input = pd.DataFrame([data])

        # Convert numeric fields
        for col in df_input.columns:
            if col in df.select_dtypes(include=[np.number]).columns:
                df_input[col] = pd.to_numeric(df_input[col], errors="coerce").fillna(0)

        # Apply label encoding
        for col in label_encoders.keys():
            if col in df_input:
                value = df_input[col].values[0]
                if value in label_encoders[col].classes_:
                    df_input[col] = label_encoders[col].transform([value])[0]
                else:
                    print(f"Unknown category: {value}")
                    df_input[col] = 0

        df_input = df_input[FEATURE_COLUMNS]
        input_scaled = scaler.transform(df_input)

        prediction_encoded = model.predict(input_scaled)
        predicted_career = label_encoder.inverse_transform(prediction_encoded)[0]

        return jsonify({"predicted_career": predicted_career})

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
