import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load Saved Files
model = joblib.load("customer_churn_catboost.pkl")
threshold = joblib.load("optimal_threshold.pkl")
model_columns = joblib.load("model_columns.pkl")

# Page Config
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📞",
    layout="wide"
)

st.markdown("""
    <style>

    div.stButton > button:first-child {
        background-color: #16a34a;
        color: white;
        font-size: 20px;
        font-weight: bold;
        height: 3.2em;
        width: 100%;
        border-radius: 10px;
        border: none;
    }

    div.stButton > button:first-child:hover {
        background-color: #15803d;
        color: white;
    }

    </style>
""", unsafe_allow_html=True)

st.title("📞 Telecom Customer Churn Prediction")

st.markdown(
    """
        Predict whether a customer is likely to churn using a tuned CatBoost model.
    """
)

# SIDEBAR INPUTS
st.sidebar.header("Customer Information")

# Basic Info
gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"]
)

age = st.sidebar.slider(
    "Age",
    18,
    100,
    35
)

married = st.sidebar.selectbox(
    "Married",
    ["Yes", "No"]
)

dependents = st.sidebar.number_input(
    "Number of Dependents",
    min_value=0,
    max_value=10,
    value=0
)

referrals = st.sidebar.number_input(
    "Number of Referrals",
    min_value=0,
    max_value=20,
    value=0
)

tenure = st.sidebar.slider(
    "Tenure in Months",
    0,
    72,
    12
)

# Services
phone_service = st.sidebar.selectbox(
    "Phone Service",
    ["Yes", "No"]
)

multiple_lines = st.sidebar.selectbox(
    "Multiple Lines",
    ["Yes", "No", "No Phone Service"]
)

internet_service = st.sidebar.selectbox(
    "Internet Service",
    ["Yes", "No"]
)

internet_type = st.sidebar.selectbox(
    "Internet Type",
    ["DSL", "Fiber Optic", "No Internet"]
)

online_security = st.sidebar.selectbox(
    "Online Security",
    ["Yes", "No", "No Internet"]
)

online_backup = st.sidebar.selectbox(
    "Online Backup",
    ["Yes", "No", "No Internet"]
)

device_protection = st.sidebar.selectbox(
    "Device Protection Plan",
    ["Yes", "No", "No Internet"]
)

premium_support = st.sidebar.selectbox(
    "Premium Tech Support",
    ["Yes", "No", "No Internet"]
)

streaming_tv = st.sidebar.selectbox(
    "Streaming TV",
    ["Yes", "No", "No Internet"]
)

streaming_movies = st.sidebar.selectbox(
    "Streaming Movies",
    ["Yes", "No", "No Internet"]
)

streaming_music = st.sidebar.selectbox(
    "Streaming Music",
    ["Yes", "No", "No Internet"]
)

unlimited_data = st.sidebar.selectbox(
    "Unlimited Data",
    ["Yes", "No", "No Internet"]
)

# Billing
contract = st.sidebar.selectbox(
    "Contract Type",
    ["Month-to-Month", "One Year", "Two Year"]
)

paperless = st.sidebar.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

payment_method = st.sidebar.selectbox(
    "Payment Method",
    ["Bank Withdrawal", "Credit Card", "Mailed Check"]
)

offer = st.sidebar.selectbox(
    "Offer",
    ["None", "Offer A", "Offer B", "Offer C", "Offer D", "Offer E"]
)

monthly_charge = st.sidebar.number_input(
    "Monthly Charge ($)",
    min_value=0.0,
    max_value=500.0,
    value=70.0
)

avg_long_distance = st.sidebar.number_input(
    "Avg Monthly Long Distance Charges ($)",
    min_value=0.0,
    max_value=500.0,
    value=10.0
)

avg_gb_download = st.sidebar.number_input(
    "Avg Monthly GB Download",
    min_value=0.0,
    max_value=1000.0,
    value=50.0
)

total_charges = st.sidebar.number_input(
    "Total Charges ($)",
    min_value=0.0,
    max_value=50000.0,
    value=1000.0
)

extra_data_charges = st.sidebar.number_input(
    "Total Extra Data Charges ($)",
    min_value=0.0,
    max_value=10000.0,
    value=0.0
)

# PREDICT BUTTON
predict_button = st.sidebar.button(
    "Predict Churn"
)

if predict_button:
    input_data = {}

    # Numerical Features
    input_data['Gender'] = 1 if gender == "Male" else 0
    input_data['Age'] = age
    input_data['Married'] = 1 if married == "Yes" else 0
    input_data['Number of Dependents'] = dependents
    input_data['Number of Referrals'] = referrals
    input_data['Tenure in Months'] = tenure
    input_data['Phone Service'] = 1 if phone_service == "Yes" else 0

    input_data['Avg Monthly Long Distance Charges'] = avg_long_distance

    input_data['Avg Monthly GB Download'] = avg_gb_download

    input_data['Paperless Billing'] = (
        1 if paperless == "Yes" else 0
    )

    input_data['Monthly Charge'] = monthly_charge

    input_data['Total Charges'] = total_charges

    input_data['Total Refunds'] = 0

    input_data['Total Extra Data Charges'] = extra_data_charges

    total_long_distance = (
        avg_long_distance * tenure
    )

    input_data['Total Long Distance Charges'] = (
        total_long_distance
    )

    total_revenue = (
        total_charges
        + extra_data_charges
        + total_long_distance
    )

    input_data['Total Revenue'] = total_revenue

    # Feature Engineering
    total_services = 0

    service_features = [
    phone_service,
    multiple_lines,
    internet_service,
    online_security,
    online_backup,
    device_protection,
    premium_support,
    streaming_tv,
    streaming_movies,
    streaming_music,
    unlimited_data
    ]

    for feature in service_features:

        if feature == "Yes":
            total_services += 1

    input_data['TotalServices'] = total_services

    input_data['AvgChargePerService'] = (
        monthly_charge / (total_services + 1)
    )

    input_data['AutoPayment'] = (
        1 if payment_method != "Mailed Check"
        else 0
    )

    # Tenure Groups
    input_data['TenureGroup_Loyal'] = (
        1 if tenure >= 48 else 0
    )

    input_data['TenureGroup_New'] = (
        1 if tenure <= 12 else 0
    )

    input_data['TenureGroup_Regular'] = (
        1 if 12 < tenure < 48 else 0
    )

    # Initialize All Columns
    for col in model_columns:
        if col not in input_data:
            input_data[col] = 0


    # One Hot Encoding

    # Offer
    if offer != "None":
        input_data[f'Offer_{offer}'] = 1

    # Multiple Lines
    if multiple_lines == "Yes":
        input_data['Multiple Lines_Yes'] = 1
    elif multiple_lines == "No Phone Service":
        input_data['Multiple Lines_No Phone Service'] = 1

    # Internet

    if internet_service == "Yes":
        input_data['Internet Service_Yes'] = 1

    if internet_type == "DSL":
        input_data['Internet Type_DSL'] = 1
    elif internet_type == "Fiber Optic":
        input_data['Internet Type_Fiber Optic'] = 1
    elif internet_type == "No Internet":
        input_data['Internet Type_No Internet'] = 1

    # Online Security
    if online_security == "Yes":
        input_data['Online Security_Yes'] = 1
    elif online_security == "No Internet":
        input_data['Online Security_No Internet'] = 1

    # Online Backup
    if online_backup == "Yes":
        input_data['Online Backup_Yes'] = 1
    elif online_backup == "No Internet":
        input_data['Online Backup_No Internet'] = 1

    # Device Protection
    if device_protection == "Yes":
        input_data['Device Protection Plan_Yes'] = 1
    elif device_protection == "No Internet":
        input_data['Device Protection Plan_No Internet'] = 1

    # Premium Support
    if premium_support == "Yes":
        input_data['Premium Tech Support_Yes'] = 1
    elif premium_support == "No Internet":
        input_data['Premium Tech Support_No Internet'] = 1

    # Streaming TV
    if streaming_tv == "Yes":
        input_data['Streaming TV_Yes'] = 1
    elif streaming_tv == "No Internet":
        input_data['Streaming TV_No Internet'] = 1

    # Streaming Movies
    if streaming_movies == "Yes":
        input_data['Streaming Movies_Yes'] = 1
    elif streaming_movies == "No Internet":
        input_data['Streaming Movies_No Internet'] = 1

    # Streaming Music
    if streaming_music == "Yes":
        input_data['Streaming Music_Yes'] = 1
    elif streaming_music == "No Internet":
        input_data['Streaming Music_No Internet'] = 1

    # Unlimited Data
    if unlimited_data == "Yes":
        input_data['Unlimited Data_Yes'] = 1
    elif unlimited_data == "No Internet":
        input_data['Unlimited Data_No Internet'] = 1

    # Contract
    if contract == "One Year":
        input_data['Contract_One Year'] = 1
    elif contract == "Two Year":
        input_data['Contract_Two Year'] = 1

    # Payment Method
    if payment_method == "Credit Card":
        input_data['Payment Method_Credit Card'] = 1
    elif payment_method == "Mailed Check":
        input_data['Payment Method_Mailed Check'] = 1

    # Create DataFrame
    input_df = pd.DataFrame([input_data])
    input_df = input_df[model_columns]

    # Prediction
    churn_probability = (
        model.predict_proba(input_df)[0][1]
    )

    prediction = (
        1 if churn_probability >= threshold else 0
    )

    # OUTPUT
    st.subheader("Prediction Result")

    st.metric(
        "Churn Probability",
        f"{churn_probability:.2%}"
    )

    st.progress(float(churn_probability))

    high_risk_threshold = threshold      
    medium_risk_threshold = 0.25

    if churn_probability >= high_risk_threshold:
        st.error("🚨 High Churn Risk")
        st.write("Immediate retention action recommended.")

    elif churn_probability >= medium_risk_threshold:
        st.warning("⚠️ Medium Churn Risk")
        st.write("Monitor closely or offer a small incentive.")

    else:
        st.success("👌 Low Churn Risk")

    # Customer Summary

    st.subheader("Customer Summary")

    summary_data = {
        "Feature": [
            "Gender",
            "Age",
            "Married",
            "Dependents",
            "Referrals",
            "Tenure (Months)",
            "Phone Service",
            "Multiple Lines",
            "Internet Service",
            "Internet Type",
            "Online Security",
            "Online Backup",
            "Device Protection",
            "Premium Tech Support",
            "Streaming TV",
            "Streaming Movies",
            "Streaming Music",
            "Unlimited Data",
            "Contract Type",
            "Paperless Billing",
            "Payment Method",
            "Offer",
            "Monthly Charge",
            "Avg Monthly Long Distance Charges",
            "Avg Monthly GB Download",
            "Total Charges",
            "Extra Data Charges"
        ],

        "Value": [
            gender,
            age,
            married,
            dependents,
            referrals,
            tenure,
            phone_service,
            multiple_lines,
            internet_service,
            internet_type,
            online_security,
            online_backup,
            device_protection,
            premium_support,
            streaming_tv,
            streaming_movies,
            streaming_music,
            unlimited_data,
            contract,
            paperless,
            payment_method,
            offer,
            f"${monthly_charge:.2f}",
            f"${avg_long_distance:.2f}",
            avg_gb_download,
            f"${total_charges:.2f}",
            f"${extra_data_charges:.2f}"
        ]
    }

    summary_df = pd.DataFrame(summary_data)

    st.dataframe(
        summary_df,
        use_container_width=True
    )


st.markdown("---")

st.markdown(
    f"""
    ### Model Information
    
    * **Model:** CatBoost Classifier
    * **ROC-AUC:** 90.54%
    * **Recall:** 79.36%
    * **Accuracy:** 82.21%
    * **Optimized Threshold:** `{threshold:.4f}`
    """, 
    unsafe_allow_html=False
)

st.markdown("---")

st.caption(
    "Built using Streamlit, CatBoost and Scikit-learn"
)