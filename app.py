import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import warnings

warnings.filterwarnings('ignore')

# 1. LOAD DATA & CACHE
@st.cache_resource
def load_assets():
    model = joblib.load("customer_churn_catboost.pkl")
    threshold = joblib.load("optimal_threshold.pkl")
    model_columns = joblib.load("model_columns.pkl")
    return model, threshold, model_columns

@st.cache_resource
def create_explainer(_model):
    return shap.TreeExplainer(_model)

model, threshold, model_columns = load_assets()
explainer = create_explainer(model)

# 2. PAGE CONFIG & STYLING
st.set_page_config(
    page_title="Telecom Churn Intelligence",
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
            }

            [data-testid="stMetricValue"] {
                font-size: 48px;
            }
        </style>
    """, unsafe_allow_html=True)

st.title("📞 Telecom Customer Churn Prediction")
st.write("Decision Support System powered by a threshold-optimized CatBoost model.")

# 3. SIDEBAR INPUTS
st.sidebar.header("Customer Information")

# 3.1. Personal Details
with st.sidebar.expander("👤 Personal Details", expanded=True):
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 18, 100, 35)
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.number_input("Number of Dependents", 0, 10, 0)
    referrals = st.number_input("Number of Referrals", 0, 20, 0)
    tenure = st.slider("Tenure in Months", 0, 72, 12)

# 3.2. Services & Usage
with st.sidebar.expander("🛠️ Services & Usage"):
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No Phone Service"])
    internet_service = st.selectbox("Internet Service", ["Yes", "No"])
    internet_type = st.selectbox("Internet Type", ["DSL", "Fiber Optic", "No Internet"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No Internet"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No Internet"])
    device_protection = st.selectbox("Device Protection Plan", ["Yes", "No", "No Internet"])
    premium_support = st.selectbox("Premium Tech Support", ["Yes", "No", "No Internet"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No Internet"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No Internet"])
    streaming_music = st.selectbox("Streaming Music", ["Yes", "No", "No Internet"])
    unlimited_data = st.selectbox("Unlimited Data", ["Yes", "No", "No Internet"])

# 3.3. Contract & Billing
with st.sidebar.expander("💳 Contract & Billing"):
    contract = st.selectbox("Contract Type", ["Month-to-Month", "One Year", "Two Year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", ["Bank Withdrawal", "Credit Card", "Mailed Check"])
    offer = st.selectbox("Offer", ["None", "Offer A", "Offer B", "Offer C", "Offer D", "Offer E"])
    monthly_charge = st.number_input("Monthly Charge ($)", 0.0, 500.0, 70.0)
    avg_long_distance = st.number_input("Avg Monthly Long Distance ($)", 0.0, 500.0, 10.0)
    avg_gb_download = st.number_input("Avg Monthly GB Download", 0.0, 1000.0, 50.0)
    total_charges = st.number_input("Total Charges ($)", 0.0, 50000.0, 1000.0)
    extra_data_charges = st.number_input("Total Extra Data Charges ($)", 0.0, 10000.0, 0.0)

predict_button = st.sidebar.button("Predict Churn")

# 4. PROCESSING & PREDICTION
if predict_button:
    input_data = {col: 0 for col in model_columns}
    
    # Direct Assignments
    input_data['Gender'] = 1 if gender == "Male" else 0
    input_data['Age'] = age
    input_data['Married'] = 1 if married == "Yes" else 0
    input_data['Number of Dependents'] = dependents
    input_data['Number of Referrals'] = referrals
    input_data['Tenure in Months'] = tenure
    input_data['Phone Service'] = 1 if phone_service == "Yes" else 0
    input_data['Avg Monthly Long Distance Charges'] = avg_long_distance
    input_data['Avg Monthly GB Download'] = avg_gb_download
    input_data['Paperless Billing'] = 1 if paperless == "Yes" else 0
    input_data['Monthly Charge'] = monthly_charge
    input_data['Total Charges'] = total_charges
    input_data['Total Refunds'] = 0
    input_data['Total Extra Data Charges'] = extra_data_charges
    
    # Calculation Features
    total_long_distance = avg_long_distance * tenure
    input_data['Total Long Distance Charges'] = total_long_distance
    input_data['Total Revenue'] = total_charges + extra_data_charges + total_long_distance
    
    # Total Services Feature
    service_list = [phone_service, multiple_lines, internet_service, online_security, 
                    online_backup, device_protection, premium_support, streaming_tv, 
                    streaming_movies, streaming_music, unlimited_data]
    total_services = sum(1 for s in service_list if s == "Yes")
    input_data['TotalServices'] = total_services
    input_data['AvgChargePerService'] = monthly_charge / (total_services + 1)
    input_data['AutoPayment'] = 1 if payment_method != "Mailed Check" else 0
    
    # Tenure Groups
    input_data['TenureGroup_Loyal'] = 1 if tenure >= 48 else 0
    input_data['TenureGroup_New'] = 1 if tenure <= 12 else 0
    input_data['TenureGroup_Regular'] = 1 if 12 < tenure < 48 else 0

    # One-Hot Encoding Logic
    if offer != "None": input_data[f'Offer_{offer}'] = 1
    if multiple_lines == "Yes": input_data['Multiple Lines_Yes'] = 1
    elif multiple_lines == "No Phone Service": input_data['Multiple Lines_No Phone Service'] = 1
    
    if internet_service == "Yes": input_data['Internet Service_Yes'] = 1
    if internet_type in ["DSL", "Fiber Optic", "No Internet"]:
        input_data[f'Internet Type_{internet_type}'] = 1

    for feat_name, val in [("Online Security", online_security), ("Online Backup", online_backup),
                           ("Device Protection Plan", device_protection), ("Premium Tech Support", premium_support),
                           ("Streaming TV", streaming_tv), ("Streaming Movies", streaming_movies),
                           ("Streaming Music", streaming_music), ("Unlimited Data", unlimited_data)]:
        if val == "Yes": input_data[f'{feat_name}_Yes'] = 1
        elif val == "No Internet": input_data[f'{feat_name}_No Internet'] = 1

    if contract == "One Year": input_data['Contract_One Year'] = 1
    elif contract == "Two Year": input_data['Contract_Two Year'] = 1

    if payment_method == "Credit Card": input_data['Payment Method_Credit Card'] = 1
    elif payment_method == "Mailed Check": input_data['Payment Method_Mailed Check'] = 1

    input_df = pd.DataFrame([input_data])[model_columns]
    churn_probability = model.predict_proba(input_df)[0][1]

    # 5. OUTPUT DISPLAY
    st.subheader("Prediction Result")
    
    st.metric("Churn Probability", f"{churn_probability:.2%}")
    st.progress(float(churn_probability))

    if churn_probability >= threshold:
        st.error(f"🚨 High Churn Risk (Probability > {threshold:.2%})")
        st.markdown("**Recommended Next Steps:**")
        
        if contract == "Month-to-Month":
            st.info("💡 **Action:** Pitch a 1-year or 2-year contract with a small sign-up discount to lock in commitment.")
        
        if referrals == 0:
            st.info("💡 **Action:** Invite them to a referral program. Getting them to invite friends is our best way to build loyalty.")
        
        if monthly_charge > 70:
            st.info("💡 **Action:** Review their billing. The current rate is a heavy churn driver, so a tailored discount or alternative plan could save them.")
        
        if dependents == 0:
            st.info("💡 **Action:** Suggest an account upgrade. Single users leave easily—cross-selling a multi-user or family plan keeps them around longer.")

    elif churn_probability >= (threshold * 0.7):
        st.warning("⚠️ Medium Churn Risk")
        st.write("This customer is showing early warning signs. Keep an eye on their recent usage trends.")
        
    else:
        st.success("👌 Low Churn Risk")
        st.write("This account looks healthy and stable.")

    st.divider()

    # 6. SHAP EXPLANATIONS
    st.subheader("🔍 Why This Prediction?")
    
    shap_values = explainer.shap_values(input_df)
    shap_df = pd.DataFrame({'Feature': input_df.columns, 'SHAP Value': shap_values[0]})

    positive_features = shap_df.sort_values(by='SHAP Value', ascending=False).head(3)
    negative_features = shap_df.sort_values(by='SHAP Value', ascending=True).head(3)

    def get_explanation(feature, shap_val, value_dict):
        """
        Translates SHAP values into business insights based on the 
        actual behavior observed in the SHAP beeswarm summary plot.
        """
        is_positive = shap_val > 0
        
        # Mapping dictionary based on your SHAP summary plots
        explanations = {
            'Number of Referrals': {
                'pos': f"Having few or no referrals ({value_dict['Number of Referrals']}) strongly flags them as a churn risk.",
                'neg': "They actively refer others, which is a fantastic sign of a loyal promoter."
            },
            'Contract_Two Year': {
                'pos': "Not having a long-term (2-year) contract leaves this account highly unprotected.",
                'neg': "The solid commitment of a 2-year contract is the biggest reason they are staying."
            },
            'Contract_One Year': {
                'pos': "Without a multi-year agreement, the customer has less reason to stay long-term.",
                'neg': "The baseline stability of a 1-year contract is actively keeping this account secure."
            },
            'Tenure in Months': {
                'pos': f"They haven't been with us very long ({value_dict['Tenure in Months']} months), so they haven't built strong brand habits yet.",
                'neg': "Their extensive history with us makes them an incredibly stable, long-term customer."
            },
            'Monthly Charge': {
                'pos': f"Their high monthly bill (${value_dict['Monthly Charge']:.2f}) is creating heavy price frustration.",
                'neg': "They are getting a highly competitive monthly rate, which makes them want to stay."
            },
            'AvgChargePerService': {
                'pos': "The average cost per individual service feels too high, lowering their perceived value.",
                'neg': "They are getting excellent value across their bundled services, keeping them happy."
            },
            'Paperless Billing': {
                'pos': "Our data shows paperless billing users in this segment have a noticeably higher tendency to leave.",
                'neg': "Their traditional billing setup aligns perfectly with our most stable customer habits."
            },
            'Number of Dependents': {
                'pos': "Single-user accounts or those without listed dependents find it much easier to walk away.",
                'neg': "Having family or multiple dependents tied to the account adds strong roots to their plan."
            },
            'Internet Type_Fiber Optic': {
                'pos': "Fiber Optic subscribers are leaving at higher rates, likely due to pricing or local competition.",
                'neg': "Our Fiber Optic service quality is keeping them happy and loyal."
            },
            'TenureGroup_New': {
                'pos': "Being in their very first year makes them highly susceptible to early-stage cancellation.",
                'neg': "Crossing the critical 12-month milestone has made them a much safer, more secure account."
            },
            'Payment Method_Credit Card': {
                'pos': "Relying on manual or non-automated credit card payments shows a slight tie to customer drop-off.",
                'neg': "Having automated credit card payments keeps their account clear and consistently active."
            }
        }
        
        if feature in explanations:
            return explanations[feature]['pos'] if is_positive else explanations[feature]['neg']
        
        readable_name = feature.replace("_", " ").replace("Yes", "").strip()
        
        if "Yes" in feature or "Internet Service" in feature:
            if is_positive:
                return f"The current setup or lack of {readable_name} is pushing this customer toward leaving."
            else:
                return f"Active use of {readable_name} is one of the main reasons they are staying."

        direction = "is driving up their churn risk" if is_positive else "is helping keep this account stable"
        return f"The customer's {readable_name} {direction}."

    exp_col1, exp_col2 = st.columns(2)
    
    with exp_col1:
        st.markdown("### ⚠️ Factors Increasing Churn Risk")

        for _, row in positive_features.iterrows():
            explanation = get_explanation(
                row['Feature'], 
                row['SHAP Value'], 
                input_data
            )

            if explanation and row['SHAP Value'] > 0:
                st.warning(explanation)

    with exp_col2:
        st.markdown("### ✅ Factors Supporting Retention")

        for _, row in negative_features.iterrows():
            explanation = get_explanation(
                row['Feature'], 
                row['SHAP Value'], 
                input_data
            )

            if explanation and row['SHAP Value'] < 0:
                st.success(explanation)

    st.divider()

    # 7. CUSTOMER SUMMARY & TECH SPECS
    with st.expander("Customer Summary"):
        summary_data = {
            "Feature": [
                "Gender", "Age", "Marital Status", "Dependents", "Referrals", "Tenure",
                "Phone Service", "Multiple Lines", "Internet Service", "Internet Type",
                "Online Security", "Online Backup", "Device Protection", "Premium Tech Support",
                "Streaming TV", "Streaming Movies", "Streaming Music", "Unlimited Data",
                "Contract Type", "Paperless Billing", "Payment Method", "Offer",
                "Monthly Charge", "Total Charges", "Total Services"
            ],
            "Value": [
                gender, age, married, dependents, referrals, f"{tenure} months",
                phone_service, multiple_lines, internet_service, internet_type,
                online_security, online_backup, device_protection, premium_support,
                streaming_tv, streaming_movies, streaming_music, unlimited_data,
                contract, paperless, payment_method, offer,
                f"${monthly_charge:.2f}", f"${total_charges:.2f}", total_services
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df["Value"] = (summary_df["Value"].astype(str))
        st.dataframe(summary_df, width='stretch', hide_index=True)

with st.expander("View Model Technical Specifications"):
    st.markdown(f"""
    * **Model:** CatBoost Classifier
    * **ROC-AUC:** 90.54%
    * **Recall:** 79.36%
    * **Accuracy:** 82.21%
    * **Optimized Threshold:** `{threshold:.4f}`
    * **Strategy:** Threshold-moving to prioritize early detection (Recall).
    """)

st.caption("Powered by CatBoost, SHAP, and Streamlit")