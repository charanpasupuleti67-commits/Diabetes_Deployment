{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d77d6d17-a963-4ec4-893d-990da204e22a",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import joblib\n",
    "import numpy as np\n",
    "\n",
    "# Load model and scaler\n",
    "model = joblib.load(\"model_1.pkl\")\n",
    "scaler = joblib.load(\"scaler_1.pkl\")\n",
    "\n",
    "st.set_page_config(page_title=\"Diabetes Prediction\", page_icon=\"💊\")\n",
    "\n",
    "st.title(\"💊 Diabetes Prediction\")\n",
    "\n",
    "st.write(\"Enter the feature values below to predict the diagnosis.\")\n",
    "\n",
    "# User Inputs\n",
    "Pregnancies = st.number_input(\"Pregnancies\", min_value=0.0, format=\"%.4f\")\n",
    "Glucose = st.number_input(\"Glucose\", min_value=0.0, format=\"%.4f\")\n",
    "BloodPressure = st.number_input(\"BloodPressure\", min_value=0.0, format=\"%.4f\")\n",
    "SkinThickness = st.number_input(\"SkinThickness\", min_value=0.0, format=\"%.4f\")\n",
    "Insulin = st.number_input(\"Insulin\", min_value=0.0, format=\"%.4f\")\n",
    "BMI = st.number_input(\"BMI\", min_value=0.0, format=\"%.4f\")\n",
    "DiabetesPedigreeFunction = st.number_input(\"DiabetesPedigreeFunction\", min_value=0.0, format=\"%.4f\")\n",
    "Age = st.number_input(\"Age\", min_value=0.0, format=\"%.4f\")\n",
    "\n",
    "if st.button(\"Predict Diabetes\"):\n",
    "\n",
    "    input_data = np.array([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])\n",
    "\n",
    "    # Scale the input\n",
    "    input_scaled = scaler.transform(input_data)\n",
    "\n",
    "    # Predict\n",
    "    prediction = model.predict(input_scaled)\n",
    "\n",
    "    # Display Result\n",
    "    if prediction[0] == 1:\n",
    "        st.error(\"🔴 Diagnosis: Malignant\")\n",
    "    else:\n",
    "        st.success(\"🟢 Diagnosis: Benign\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "88531002-08b6-44c1-850e-2be9e747ecb8",
   "metadata": {},
   "outputs": [],
   "source": [
    "Pregnancies\tGlucose\tBloodPressure\tSkinThickness\tInsulin\tBMI\tDiabetesPedigreeFunction\tAge"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3e19f2a4-971d-4a89-987a-b26db491f2be",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ce572c12-7a87-4ebe-84ab-514ba7be7808",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
