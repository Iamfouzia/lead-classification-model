import streamlit as st
import pandas as pd
import joblib

# Load the saved model and preprocessing tools
model = joblib.load('lead_priority_model.pkl')
scaler = joblib.load('scaler.pkl')
le_website = joblib.load('le_website.pkl')
le_automation = joblib.load('le_automation.pkl')
le_priority = joblib.load('le_priority.pkl')

st.title('Lead Priority Classifier (ML Model)')

# Load and clean the data the same way as during training
df = pd.read_excel('leads.xlsx')
df['Reviews Count'] = (
    df['Reviews Count']
    .astype(str)
    .str.replace(',', '', regex=False)
    .str.replace('+', '', regex=False)
    .astype(int)
)
df['Has_Social_Media'] = df['Social Media Links'].notna().astype(int)

# Encode using the same encoders used during training
df['Website_Available_Encoded'] = le_website.transform(df['Website Available'])
df['Automation_Status_Encoded'] = le_automation.transform(df['Automation Status'])

# Prepare features and scale them
X = df[['Website_Available_Encoded', 'Automation_Status_Encoded',
        'Google Rating', 'Reviews Count', 'Has_Social_Media']]
X_scaled = scaler.transform(X)

# Predict priority for all leads
predictions_encoded = model.predict(X_scaled)
df['Predicted Priority'] = le_priority.inverse_transform(predictions_encoded)

st.subheader('All Leads: Actual vs Predicted Priority')
st.dataframe(
    df[['Business Name', 'City', 'Lead Priority', 'Predicted Priority']],
    use_container_width=True
)