import streamlit as st
import pandas as pd
import joblib

# Load the trained model and StandardScaler
model = joblib.load("model_1.pkl")
scaler = joblib.load("scaler_1.pkl")

st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺")
st.title("Diabetes Prediction App")
st.write("Enter the patient's details below and click **Predict**.")

# Keep the same feature order used when training the model
feature_names = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

with st.form("prediction_form"):
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.number_input("Glucose", min_value=0.0, max_value=300.0, value=120.0)
    blood_pressure = st.number_input("BloodPressure", min_value=0.0, max_value=200.0, value=70.0)
    skin_thickness = st.number_input("SkinThickness", min_value=0.0, max_value=100.0, value=20.0)
    insulin = st.number_input("Insulin", min_value=0.0, max_value=900.0, value=80.0)
    bmi = st.number_input("BMI", min_value=0.0, max_value=80.0, value=25.0)
    dpf = st.number_input(
        "DiabetesPedigreeFunction", min_value=0.0, max_value=3.0,
        value=0.5, step=0.01
    )
    age = st.number_input("Age", min_value=1, max_value=120, value=30)

    submitted = st.form_submit_button("Predict")

if submitted:
    input_data = pd.DataFrame(
        [[pregnancies, glucose, blood_pressure, skin_thickness,
          insulin, bmi, dpf, age]],
        columns=feature_names
    )

    # Apply the same StandardScaler saved during model training
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.error("The model predicts: Diabetes")
    else:
        st.success("The model predicts: No Diabetes")

    # Show the model's estimated probability for the positive class
    probability = model.predict_proba(input_scaled)[0][1]
    st.write(f"Predicted probability of diabetes: **{probability:.2%}**")

st.caption(
    "Educational demonstration only. This prediction is not a medical diagnosis."
)
