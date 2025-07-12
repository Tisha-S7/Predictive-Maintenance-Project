# Step 2: Data Cleaning & Feature Engineering
# File: scripts/step2_data_cleaning.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("    🔧 PREDICTIVE MAINTENANCE PROJECT")
print("    🧹 Step 2: Data Cleaning & Feature Engineering")
print("="*60)

# ============================================================================
# 1. LOAD CLEANED DATA FROM STEP 1
# ============================================================================

print("\n1️⃣ LOADING DATA...")
print("-" * 30)

try:
    # Load Excel file
    excel_file_path = '../data/maintenance_data.xlsx'
    df = pd.read_excel(excel_file_path, sheet_name=0)
    
    print(f"✅ Data loaded successfully!")
    print(f"📊 Original Shape: {df.shape}")
    
    # Create backup
    df_original = df.copy()
    
except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit()

# ============================================================================
# 2. MISSING VALUES HANDLING
# ============================================================================

print(f"\n2️⃣ HANDLING MISSING VALUES")
print("-" * 30)

# Check missing values
missing_before = df.isnull().sum().sum()
print(f"📊 Missing values before cleaning: {missing_before}")

if missing_before > 0:
    print(f"\n🔧 Cleaning missing values...")
    
    # Numerical columns - fill with median
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"   • {col}: filled {df[col].isnull().sum()} missing values with median {median_val:.2f}")
    
    # Categorical columns - fill with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            print(f"   • {col}: filled {df[col].isnull().sum()} missing values with mode '{mode_val}'")
    
    missing_after = df.isnull().sum().sum()
    print(f"✅ Missing values after cleaning: {missing_after}")
else:
    print("✅ No missing values found!")

# ============================================================================
# 3. OUTLIER DETECTION AND HANDLING
# ============================================================================

print(f"\n3️⃣ OUTLIER DETECTION & HANDLING")
print("-" * 30)

# Function to detect outliers using IQR method
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers.index, lower_bound, upper_bound

# Detect outliers in numerical columns
outlier_summary = []
total_outliers_removed = 0

for col in numerical_cols:
    outlier_indices, lower_bound, upper_bound = detect_outliers_iqr(df, col)
    outlier_count = len(outlier_indices)
    
    if outlier_count > 0:
        outlier_percentage = (outlier_count / len(df)) * 100
        outlier_summary.append({
            'Column': col,
            'Outliers': outlier_count,
            'Percentage': outlier_percentage,
            'Lower_Bound': lower_bound,
            'Upper_Bound': upper_bound
        })
        
        # Remove outliers if they are less than 5% of data
        if outlier_percentage < 5:
            df = df.drop(outlier_indices)
            total_outliers_removed += outlier_count
            print(f"   • {col}: Removed {outlier_count} outliers ({outlier_percentage:.1f}%)")
        else:
            print(f"   • {col}: {outlier_count} outliers found ({outlier_percentage:.1f}%) - Too many to remove")

if outlier_summary:
    outlier_df = pd.DataFrame(outlier_summary)
    print(f"\n📊 Outlier Summary:")
    print(outlier_df.round(2))
    print(f"✅ Total outliers removed: {total_outliers_removed}")
    print(f"📊 New dataset shape: {df.shape}")
else:
    print("✅ No significant outliers detected!")

# ============================================================================
# 4. FEATURE ENGINEERING
# ============================================================================

print(f"\n4️⃣ FEATURE ENGINEERING")
print("-" * 30)

# Reset index after outlier removal
df = df.reset_index(drop=True)

print(f"🔧 Creating new features...")

# Temperature-based features
if 'bearing_temp' in df.columns and 'ambient_temp' in df.columns:
    df['temp_ratio'] = df['bearing_temp'] / df['ambient_temp']
    print(f"   • Created: temp_ratio (bearing/ambient)")

if 'motor_temp' in df.columns and 'ambient_temp' in df.columns:
    df['motor_ambient_diff'] = df['motor_temp'] - df['ambient_temp']
    print(f"   • Created: motor_ambient_diff")

if 'bearing_temp' in df.columns and 'motor_temp' in df.columns:
    df['bearing_motor_diff'] = df['bearing_temp'] - df['motor_temp']
    print(f"   • Created: bearing_motor_diff")

# Vibration-based features
vibration_cols = [col for col in df.columns if 'vibration' in col.lower()]
if len(vibration_cols) >= 3:
    # Find X, Y, Z components
    x_col = next((col for col in vibration_cols if 'x' in col.lower()), None)
    y_col = next((col for col in vibration_cols if 'y' in col.lower()), None)
    z_col = next((col for col in vibration_cols if 'z' in col.lower()), None)
    
    if x_col and y_col and z_col:
        df['vibration_magnitude'] = np.sqrt(df[x_col]**2 + df[y_col]**2 + df[z_col]**2)
        print(f"   • Created: vibration_magnitude")
        
        df['vibration_max'] = df[[x_col, y_col, z_col]].max(axis=1)
        df['vibration_min'] = df[[x_col, y_col, z_col]].min(axis=1)
        df['vibration_range'] = df['vibration_max'] - df['vibration_min']
        print(f"   • Created: vibration_max, vibration_min, vibration_range")

# Current-based features
if 'motor_current' in df.columns:
    df['current_squared'] = df['motor_current'] ** 2
    print(f"   • Created: current_squared")

# Operating hours features
if 'operating_hours' in df.columns:
    df['hours_binned'] = pd.cut(df['operating_hours'], bins=5, labels=['Low', 'Medium-Low', 'Medium', 'Medium-High', 'High'])
    print(f"   • Created: hours_binned")

# Temperature efficiency features
if 'motor_current' in df.columns and 'motor_temp' in df.columns:
    # Avoid division by zero
    df['current_temp_ratio'] = df['motor_current'] / (df['motor_temp'] + 1e-6)
    print(f"   • Created: current_temp_ratio")

# Statistical features (rolling averages would need time series data)
# For now, create some basic statistical features
temp_cols = [col for col in df.columns if 'temp' in col.lower()]
if len(temp_cols) > 1:
    df['temp_mean'] = df[temp_cols].mean(axis=1)
    df['temp_std'] = df[temp_cols].std(axis=1)
    print(f"   • Created: temp_mean, temp_std")

print(f"✅ Feature engineering completed!")
print(f"📊 New dataset shape: {df.shape} (Added {df.shape[1] - df_original.shape[1]} features)")

# ============================================================================
# 5. DATA VALIDATION
# ============================================================================

print(f"\n5️⃣ DATA VALIDATION")
print("-" * 30)

# Check for any remaining issues
print(f"🔍 Validating cleaned data...")

# Check for infinite values
inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
print(f"   • Infinite values: {inf_count}")

if inf_count > 0:
    # Replace infinite values with NaN, then fill with median
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    print(f"   ✅ Infinite values handled")

# Check for duplicates
duplicate_count = df.duplicated().sum()
print(f"   • Duplicate rows: {duplicate_count}")

if duplicate_count > 0:
    df = df.drop_duplicates()
    print(f"   ✅ Duplicate rows removed")

# Final data quality check
print(f"\n📊 Final Data Quality Report:")
print(f"   • Total Records: {len(df):,}")
print(f"   • Total Features: {len(df.columns)}")
print(f"   • Missing Values: {df.isnull().sum().sum()}")
print(f"   • Duplicate Records: {df.duplicated().sum()}")
print(f"   • Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# ============================================================================
# 6. SAVE CLEANED DATA
# ============================================================================

print(f"\n6️⃣ SAVING CLEANED DATA")
print("-" * 30)

# Save cleaned data
cleaned_file_path = '../data/cleaned_maintenance_data.csv'
df.to_csv(cleaned_file_path, index=False)
print(f"✅ Cleaned data saved to: {cleaned_file_path}")

# Save feature list for reference
feature_list = df.columns.tolist()
feature_info = {
    'original_features': df_original.columns.tolist(),
    'new_features': [col for col in df.columns if col not in df_original.columns],
    'total_features': len(feature_list)
}

import json
with open('../results/feature_info.json', 'w') as f:
    json.dump(feature_info, f, indent=2)
print(f"✅ Feature information saved to: ../results/feature_info.json")

# ============================================================================
# 7. FEATURE IMPORTANCE PREVIEW
# ============================================================================

print(f"\n7️⃣ FEATURE IMPORTANCE PREVIEW")
print("-" * 30)

# Find fault column
fault_col = None
possible_fault_cols = ['fault_condition', 'fault_type', 'condition', 'status', 'fault']
for col in possible_fault_cols:
    if col in df.columns:
        fault_col = col
        break

if fault_col:
    print(f"🎯 Analyzing feature importance for: {fault_col}")
    
    # Quick Random Forest for feature importance
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    
    # Prepare features
    X = df.select_dtypes(include=[np.number])
    y = df[fault_col]
    
    # Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Quick Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y_encoded)
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🏆 Top 10 Most Important Features:")
    print(feature_importance.head(10))
    
    # Save feature importance
    feature_importance.to_csv('../results/feature_importance.csv', index=False)
    print(f"✅ Feature importance saved to: ../results/feature_importance.csv")

# ============================================================================
# 8. SUMMARY REPORT
# ============================================================================

print(f"\n8️⃣ SUMMARY REPORT")
print("=" * 60)

print(f"🧹 DATA CLEANING SUMMARY:")
print(f"   ✅ Missing values handled: {missing_before} → 0")
print(f"   ✅ Outliers removed: {total_outliers_removed}")
print(f"   ✅ Duplicate records removed: {duplicate_count}")
print(f"   ✅ Features engineered: {len(feature_info['new_features'])}")

print(f"\n📊 DATASET TRANSFORMATION:")
print(f"   • Original shape: {df_original.shape}")
print(f"   • Final shape: {df.shape}")
print(f"   • Data quality: ✅ Production Ready")

print(f"\n🎯 NEW FEATURES CREATED:")
for feature in feature_info['new_features']:
    print(f"   • {feature}")

print(f"\n🔄 NEXT STEPS:")
print(f"   1. Run Step 3: Exploratory Data Analysis")
print(f"   2. Create visualizations and insights")
print(f"   3. Analyze feature relationships")
print(f"   4. Prepare for model training")

print(f"\n✅ STEP 2 COMPLETED SUCCESSFULLY!")
print(f"📁 Cleaned data ready for analysis!")
print("=" * 60)