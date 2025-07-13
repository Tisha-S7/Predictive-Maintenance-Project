# File: scripts/step4_modeling.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("    🤖 STEP 4: MODEL TRAINING & EVALUATION")
print("="*60)

# Load the data
df = pd.read_excel('data/maintenance_data.xlsx')

# Encode categorical variables
from sklearn.preprocessing import LabelEncoder
cat_cols = df.select_dtypes(include='object').columns
if len(cat_cols) > 0:
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

# Select features and target
# Replace 'maintenance_required' with your actual target column name
if 'maintenance_required' in df.columns:
    X = df.drop(['maintenance_required'], axis=1)
    y = df['maintenance_required']
else:
    print("❌ 'maintenance_required' column not found.")
    exit()

# Split data
print("📊 Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
print("🚀 Training Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
print("📈 Evaluating model...")
y_pred = model.predict(X_test)

print("\n🔹 Accuracy:", accuracy_score(y_test, y_pred))
print("\n🔹 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n🔹 Classification Report:\n", classification_report(y_test, y_pred))

# Save model
joblib.dump(model, 'outputs/maintenance_model.pkl')
print("\n✅ Model saved as 'maintenance_model.pkl' in outputs folder.")

print(df['maintenance_required'].value_counts())
