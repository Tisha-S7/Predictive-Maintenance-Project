import pandas as pd
import joblib

# Load the saved model
model = joblib.load('outputs/maintenance_model.pkl')
print("✅ Model loaded!")

# Create sample input (dictionary doesn't need to be ordered)
sample_input = {
    'ambient_temp': [78.5],
    'bearing_temp': [65.3],
    'motor_temp': [88.7],
    'vibration_x': [0.03],
    'vibration_y': [0.05],
    'vibration_z': [0.04],
    'vibration_rms': [0.06],
    'motor_current': [14.5],
    'temp_diff': [23.4],
    'operating_hours': [1234],
    'fault_condition': [1],
    'fault_code': [3],
    'load_percentage': [80.2],
    'machine_id': [2],
    'rpm': [1450],
    'timestamp': [0]
}

# Convert to DataFrame
df = pd.DataFrame(sample_input)

# ✅ Reorder columns to match model training order
df = df[model.feature_names_in_]

# Predict
prediction = model.predict(df)
print("🔍 Prediction result:", prediction)
print("🛠️ Maintenance Required:" , "Yes" if prediction[0] == 1 else "No")
