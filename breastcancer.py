import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Tumor Classification", layout="centered")

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

st.title("Tumor Type Classifier")
st.markdown("Provide tumor characteristics to predict if it's **Malignant** or **Benign**.")

features = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness']
input_data = []

for feat in features:
    val = st.number_input(f"{feat}", min_value=0.0, value=0.0, format="%.4f")
    input_data.append(val)

if st.button("Predict Tumor Type"):
    while len(input_data) < 30:
        input_data.append(0.0)
    user_input = np.array(input_data).reshape(1, -1)
    user_input = scaler.transform(user_input)
    pred = model.predict(user_input)
    if pred[0] == 0:
        st.error("Prediction: The tumor is **Malignant** (Cancerous).")
    else:
        st.success("Prediction: The tumor is **Benign** (Non-Cancerous).")
