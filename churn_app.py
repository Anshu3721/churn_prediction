import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load trained model
model = joblib.load("best_churn_model.pkl")

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
st.title("📉 Customer Churn Prediction App")
st.markdown("Enter customer information below to predict if they are likely to churn.")

# Layout Input Fields: 4 per row

# Row 1
col1, col2, col3, col4 = st.columns(4)
with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
with col2:
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
with col3:
    Partner = st.selectbox("Has Partner?", ["Yes", "No"])
with col4:
    Dependents = st.selectbox("Has Dependents?", ["Yes", "No"])

# Row 2
col1, col2, col3, col4 = st.columns(4)
with col1:
    tenure = st.slider("Tenure (months)", 0, 72, 12)
with col2:
    PhoneService = st.selectbox("Phone Service?", ["Yes", "No"])
with col3:
    MultipleLines = st.selectbox("Multiple Lines?", ["Yes", "No"])
with col4:
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

# Row 3
col1, col2, col3, col4 = st.columns(4)
with col1:
    OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
with col2:
    OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
with col3:
    DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
with col4:
    TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])

# Row 4
col1, col2, col3, col4 = st.columns(4)
with col1:
    StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
with col2:
    StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
with col3:
    Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
with col4:
    PaperlessBilling = st.selectbox("Paperless Billing?", ["Yes", "No"])

# Row 5
col1, col2, col3 = st.columns(3)
with col1:
    PaymentMethod = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
with col2:
    MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, max_value=150.0, value=70.0)
with col3:
    TotalCharges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=3000.0)

# Predict button
if st.button("Predict Churn"):
    # Map binary features
    input_dict = {
        "gender": 1 if gender == "Female" else 0,
        "SeniorCitizen": SeniorCitizen,
        "Partner": 1 if Partner == "Yes" else 0,
        "Dependents": 1 if Dependents == "Yes" else 0,
        "tenure": tenure,
        "PhoneService": 1 if PhoneService == "Yes" else 0,
        "MultipleLines": 1 if MultipleLines == "Yes" else 0,
        "PaperlessBilling": 1 if PaperlessBilling == "Yes" else 0,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }

    # One-hot features
    categories = {
        "InternetService_Fiber optic": InternetService == "Fiber optic",
        "InternetService_No": InternetService == "No",
        "OnlineSecurity_No": OnlineSecurity == "No",
        "OnlineSecurity_No internet service": OnlineSecurity == "No internet service",
        "OnlineBackup_No": OnlineBackup == "No",
        "OnlineBackup_No internet service": OnlineBackup == "No internet service",
        "DeviceProtection_No": DeviceProtection == "No",
        "DeviceProtection_No internet service": DeviceProtection == "No internet service",
        "TechSupport_No": TechSupport == "No",
        "TechSupport_No internet service": TechSupport == "No internet service",
        "StreamingTV_No": StreamingTV == "No",
        "StreamingTV_No internet service": StreamingTV == "No internet service",
        "StreamingMovies_No": StreamingMovies == "No",
        "StreamingMovies_No internet service": StreamingMovies == "No internet service",
        "Contract_One year": Contract == "One year",
        "Contract_Two year": Contract == "Two year",
        "PaymentMethod_Credit card (automatic)": PaymentMethod == "Credit card (automatic)",
        "PaymentMethod_Electronic check": PaymentMethod == "Electronic check",
        "PaymentMethod_Mailed check": PaymentMethod == "Mailed check"
    }

    input_dict.update(categories)

    # Ensure all required features are present
    model_features = model.feature_names_in_
    for col in model_features:
        input_dict.setdefault(col, 0)

    # Create DataFrame
    input_df = pd.DataFrame([input_dict])[list(model_features)]

    # Predict
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    # Output
    st.subheader("🔍 Prediction Result")
    if prediction == 1:
        st.error("⚠️ This customer is likely to churn!")
    else:
        st.success("✅ This customer is likely to stay.")
    st.write(f"**Churn Probability:** `{probability:.2%}`")