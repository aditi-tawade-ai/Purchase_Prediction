import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("knn.pkl")
scaler = joblib.load("scaler.pkl")

# Title
st.title("Social Network Purchase Prediction")

st.write("Enter the person's details:")

# Gender
gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

# Age
age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

# Salary
salary = st.number_input(
    "Estimated Salary",
    min_value=0,
    max_value=1000000,
    value=50000
)

# Gender encoding
if gender == "Male":
    gender_encoded = 0
else:
    gender_encoded = 1

# Prediction
if st.button("Predict"):

    # Same order as training:
    # Gender, Age, EstimatedSalary
    input_data = np.array([
        [gender_encoded, age, salary]
    ])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.success("The person is likely to PURCHASE the product.")
    else:
        st.error("The person is NOT likely to purchase the product.")