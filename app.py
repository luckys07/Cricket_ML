# Assingment AI_ML

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

# # Load dataset
# data = pd.read_csv("cricket_data.csv")

# X = data[["runs", "overs", "wickets"]]
# y = data["final_score"]

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="Cricket Score Predictor",
    page_icon="🏏",
    layout="centered"
)

# ----------------- CUSTOM BACKGROUND -----------------
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
     background-image: url("bg.jpg");

    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}

[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.8);
}

.stButton>button {
    background-color: #FF9933;
    color: white;
    border-radius: 10px;
    height: 50px;
    font-size: 18px;
    font-weight: bold;
    box-shadow: 2px 2px 8px #444;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)


# ----------------- DATA + MODEL -----------------
data = pd.read_csv("cricket_data.csv")

X = data[["runs", "overs", "wickets"]]
y = data["final_score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
# # Train model
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Streamlit UI
# st.title("🏏 Cricket Score Prediction App")
# st.write("Predict the final score of a cricket team based on current stats.")

# # Inputs
# runs = st.number_input("Current Runs", min_value=0, max_value=300, value=50)
# overs = st.number_input("Overs Completed", min_value=0, max_value=20, value=10)
# wickets = st.number_input("Wickets Fallen", min_value=0, max_value=10, value=2)
# ----------------- UI -----------------
st.title("🏏 Cricket Score Prediction App")
st.markdown("### 🎯 Predict the **Final Score** of a cricket team based on the current match situation!")

st.markdown("⚡ Enter live match stats below:")

# Inputs
col1, col2, col3 = st.columns(3)
with col1:
    runs = st.number_input("🏃 Current Runs", min_value=0, max_value=300, value=50)
with col2:
    overs = st.number_input("⏱ Overs Completed", min_value=0, max_value=20, value=10)
with col3:
    wickets = st.number_input("❌ Wickets Fallen", min_value=0, max_value=10, value=2)

# Prediction Button

if st.button(" 🔮 Predict Final Score"):
    input_data = np.array([[runs, overs, wickets]])
    prediction = model.predict(input_data)[0]
    # st.success(f"🎯 Predicted Final Score: {int(prediction)} runs")
    st.success(f"🏆 Predicted Final Score: **{int(prediction)} runs**")
    st.balloons()
