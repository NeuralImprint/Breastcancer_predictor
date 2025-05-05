# Importing libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Function to take user input and predict
def predict_from_user_input():
    print("Please enter the following details about the tumor:")
    
    # List of features (you can show all 30 features, but for simplicity, let's take a few important ones)
    feature_names = [
        'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness'
    ]
    
    user_data = []
    for feature in feature_names:
        value = float(input(f"Enter {feature}: "))
        user_data.append(value)
    
    # Since model expects 30 features, we'll fill the rest with 0s
    while len(user_data) < 30:
        user_data.append(0.0)
    
    user_data = np.array(user_data).reshape(1, -1)
    
    # Scale user input (because we scaled training data)
    user_data = scaler.transform(user_data)
    
    # Predict
    prediction = model.predict(user_data)
    
    if prediction[0] == 0:
        print("\nPrediction: The tumor is Malignant (Cancerous).")
    else:
        print("\nPrediction: The tumor is Benign (Non-Cancerous).")

# Call the function
predict_from_user_input()
