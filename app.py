# Assingment AI_ML

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

# Load dataset
data = pd.read_csv("cricket_data.csv")

X = data[["runs", "overs", "wickets"]]
y = data["final_score"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit UI
st.title("🏏 Cricket Score Prediction App")
st.write("Predict the final score of a cricket team based on current stats.")

# Inputs
runs = st.number_input("Current Runs", min_value=0, max_value=300, value=50)
overs = st.number_input("Overs Completed", min_value=0, max_value=20, value=10)
wickets = st.number_input("Wickets Fallen", min_value=0, max_value=10, value=2)

if st.button("Predict Final Score"):
    input_data = np.array([[runs, overs, wickets]])
    prediction = model.predict(input_data)[0]
    st.success(f"🎯 Predicted Final Score: {int(prediction)} runs")
