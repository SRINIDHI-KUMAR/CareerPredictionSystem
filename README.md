
# Career Prediction System

This project forecasts the best career choice for users based on machine learning (Random Forest Classifier). The model is trained with a labeled career prediction data. It processes preprocessing such as label encoding and feature scaling and then learns to recognize patterns in the data in order to suggest careers.

📂Project Structure
 ├── app.py                         # Flask backend logic
 ├── train.py                       # Training script for the ML model
 ├── career_prediction_model.pkl    # Trained model file (generated after training)
 ├── label_encoder.pkl              # LabelEncoder for target variable
 ├── scaler.pkl                     # StandardScaler for numeric features
 ├
 ├── templates/
 │   ├── welcome.html               # landing page
 │   ├── index.html                 # Main page with form for prediction
 │   ├── about.html                 # About the app
 ├
 ├── static/
 │   ├── a.mp4                      # mp4 file for welcome page
 │   ├── m.mp4                      # mp4 file for index page
 ├
 ├── career_prediction_dataset.csv  # Dataset used for training/testing


 
## Installation & Setup

### 
1️⃣ Clone the Repository
```sh
git clone https://github.com/your-username/CareerPredictionSystem.git
cd CareerPredictionSystem
```
2️⃣ Create and Activate a Virtual Environment
```
For Windows:
python -m venv venv
venv\Scripts\activate


3️⃣ Install Dependencies
```
pip install scikit-learn
```

4️⃣Train the Model
```
python train.py
```

5️⃣Run the Flask App
```
python app.py
```
