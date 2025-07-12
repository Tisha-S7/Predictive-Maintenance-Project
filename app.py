import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("outputs/maintenance_model.pkl")

st.set_page_config(page_title="Predictive Maintenance", layout="centered")

# Title and description
st.markdown("<h1 style='text-align: center; color: steelblue;'>🔧 Predictive Maintenance Checker</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter sensor values to predict if maintenance is required.</p>", unsafe_allow_html=True)
st.markdown("---")

# Form layout
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        ambient_temp = st.slider("🌡️ Ambient Temperature", 60.0, 100.0, 75.0)
        bearing_temp = st.slider("🔩 Bearing Temperature", 60.0, 120.0, 80.0)
        motor_temp = st.slider("⚙️ Motor Temperature", 60.0, 150.0, 90.0)
        vibration_x = st.number_input("📈 Vibration X", value=0.02, step=0.01)
        vibration_y = st.number_input("📈 Vibration Y", value=0.03, step=0.01)
        vibration_z = st.number_input("📈 Vibration Z", value=0.04, step=0.01)
        vibration_rms = st.number_input("📊 Vibration RMS", value=0.05, step=0.01)

    with col2:
        motor_current = st.slider("⚡ Motor Current (A)", 0.0, 30.0, 15.0)
        temp_diff = st.slider("🌡️ Temperature Difference", 0.0, 100.0, 25.0)
        operating_hours = st.number_input("⏱️ Operating Hours", value=1000)
        fault_condition = st.selectbox("🛑 Fault Condition", options=[0, 1, 2])
        fault_code = st.selectbox("📟 Fault Code", options=[0, 1, 2, 3])
        load_percentage = st.slider("📦 Load Percentage", 0.0, 100.0, 75.0)
        machine_id = st.number_input("🆔 Machine ID", value=1)
        rpm = st.slider("🔁 RPM", 500, 2000, 1500)
        timestamp = st.number_input("⏰ Timestamp (Encoded)", value=0)

    submitted = st.form_submit_button("🔍 Predict Maintenance Need")

if submitted:
    # Prepare input
    input_data = pd.DataFrame([{
        'ambient_temp': ambient_temp,
        'bearing_temp': bearing_temp,
        'motor_temp': motor_temp,
        'vibration_x': vibration_x,
        'vibration_y': vibration_y,
        'vibration_z': vibration_z,
        'vibration_rms': vibration_rms,
        'motor_current': motor_current,
        'temp_diff': temp_diff,
        'operating_hours': operating_hours,
        'fault_condition': fault_condition,
        'fault_code': fault_code,
        'load_percentage': load_percentage,
        'machine_id': machine_id,
        'rpm': rpm,
        'timestamp': timestamp
    }])
    
    input_data = input_data[model.feature_names_in_]

    # Predict
    prediction = model.predict(input_data)[0]

    st.markdown("---")
    if prediction == 1:
        st.error("❗ Maintenance Required", icon="⚠️")
    else:
        st.success("✅ No Maintenance Needed", icon="✅")
