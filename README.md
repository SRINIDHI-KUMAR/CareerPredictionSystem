# 🧠 Career Prediction System

This project forecasts the best career choice for users based on machine learning (Random Forest Classifier). The model is trained with a labeled career prediction data. It processes preprocessing such as label encoding and feature scaling and then learns to recognize patterns in the data in order to suggest careers.



📂Project Structure<br>
 ├── app.py                         # Flask backend logic<br>
 ├── train.py                       # Training script for the ML model<br>
 ├── career_prediction_model.pkl    # Trained model file (generated after training)<br>
 ├── label_encoder.pkl              # LabelEncoder for target variable<br>
 ├── scaler.pkl                     # StandardScaler for numeric features<br>
 ├<br>
 ├── templates/<br>
 │   ├── welcome.html               # landing page<br>
 │   ├── index.html                 # Main page with form for prediction<br>
 │   ├── about.html                 # About the app<br>
 ├<br>
 ├── static/<br>
 │   ├── a.mp4                      # mp4 file for welcome page<br>
 │   ├── m.mp4                      # mp4 file for index page<br>
 ├<br>
 ├── career_prediction_dataset.csv  # Dataset used for training/testing<br>


 
## ⚙️ Installation & Setup

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
```

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
