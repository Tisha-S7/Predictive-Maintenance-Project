# Step 1: Excel Data Loading & Exploration
# File: scripts/step1_data_exploration.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

print("="*60)
print("    🔧 PREDICTIVE MAINTENANCE PROJECT")
print("    📊 Step 1: Data Loading & Exploration")
print("="*60)

# ============================================================================
# 1. LOAD EXCEL FILE
# ============================================================================

print("\n1️⃣ LOADING EXCEL FILE...")
print("-" * 30)

try:
    # Load Excel file - CHANGE FILE NAME HERE IF NEEDED
    excel_file_path = '../data/maintenance_data.xlsx'
    
    # Check available sheets
    excel_file = pd.ExcelFile(excel_file_path)
    print(f"📋 Available sheets: {excel_file.sheet_names}")
    
    # Load data from first sheet (change sheet name if needed)
    df = pd.read_excel(excel_file_path, sheet_name=0)  # 0 means first sheet
    
    print(f"✅ Excel file loaded successfully!")
    print(f"📊 Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
except FileNotFoundError:
    print("❌ Excel file not found!")
    print("Please ensure file is in: data/maintenance_data.xlsx")
    print("Current working directory:", os.getcwd())
    exit()
except Exception as e:
    print(f"❌ Error loading Excel file: {e}")
    exit()

# ============================================================================
# 2. BASIC DATA INFORMATION
# ============================================================================

print(f"\n2️⃣ BASIC DATA INFORMATION")
print("-" * 30)

print(f"📈 Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"💾 Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print(f"\n📑 Column Names:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:2d}. {col}")

print(f"\n📊 Data Types:")
print(df.dtypes)

print(f"\n🔍 First 5 Rows:")
print(df.head())

print(f"\n🔍 Last 5 Rows:")
print(df.tail())

# ============================================================================
# 3. MISSING VALUES ANALYSIS
# ============================================================================

print(f"\n3️⃣ MISSING VALUES ANALYSIS")
print("-" * 30)

missing_data = df.isnull().sum()
missing_percent = (missing_data / len(df)) * 100

missing_df = pd.DataFrame({
    'Missing_Count': missing_data,
    'Missing_Percentage': missing_percent.round(2)
}).sort_values('Missing_Count', ascending=False)

print("Missing Values Summary:")
print(missing_df)

total_missing = missing_df['Missing_Count'].sum()
if total_missing == 0:
    print("✅ No missing values found!")
else:
    print(f"⚠️  Total missing values: {total_missing}")
    print("🔧 Action needed: Handle missing values in next step")

# ============================================================================
# 4. STATISTICAL SUMMARY
# ============================================================================

print(f"\n4️⃣ STATISTICAL SUMMARY")
print("-" * 30)

# Numerical columns
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"📊 Numerical columns ({len(numerical_cols)}): {numerical_cols}")

print(f"\n📈 Statistical Summary:")
print(df[numerical_cols].describe().round(3))

# Categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f"\n📋 Categorical columns ({len(categorical_cols)}): {categorical_cols}")

if categorical_cols:
    print(f"\n📊 Categorical Summary:")
    for col in categorical_cols:
        print(f"\n{col}:")
        print(df[col].value_counts())

# ============================================================================
# 5. FAULT CONDITION ANALYSIS
# ============================================================================

print(f"\n5️⃣ FAULT CONDITION ANALYSIS")
print("-" * 30)

# Identify fault column (common names)
fault_col = None
possible_fault_cols = ['fault_condition', 'fault_type', 'condition', 'status', 'fault']

for col in possible_fault_cols:
    if col in df.columns:
        fault_col = col
        break

if fault_col:
    print(f"🎯 Fault column found: '{fault_col}'")
    
    fault_counts = df[fault_col].value_counts()
    fault_percent = (fault_counts / len(df)) * 100
    
    fault_analysis = pd.DataFrame({
        'Count': fault_counts,
        'Percentage': fault_percent.round(2)
    })
    
    print(f"\n📊 Fault Distribution:")
    print(fault_analysis)
    
    # Class balance analysis
    normal_cases = fault_counts.get('Normal', 0)
    total_cases = len(df)
    normal_percent = (normal_cases / total_cases) * 100
    
    print(f"\n⚖️  Class Balance Analysis:")
    print(f"   Normal cases: {normal_percent:.1f}%")
    print(f"   Fault cases: {100 - normal_percent:.1f}%")
    
    if normal_percent > 85:
        print("🚨 Highly imbalanced dataset - Need SMOTE/balancing")
    elif normal_percent > 70:
        print("⚠️  Moderately imbalanced - Consider balancing")
    else:
        print("✅ Dataset is reasonably balanced")
        
else:
    print("❌ Fault condition column not found!")
    print("Available columns:", df.columns.tolist())

# ============================================================================
# 6. CORRELATION ANALYSIS
# ============================================================================

print(f"\n6️⃣ CORRELATION ANALYSIS")
print("-" * 30)

if fault_col and len(numerical_cols) > 1:
    # Create numerical encoding for fault conditions
    from sklearn.preprocessing import LabelEncoder
    
    le = LabelEncoder()
    df_encoded = df.copy()
    df_encoded[f'{fault_col}_encoded'] = le.fit_transform(df[fault_col])
    
    # Calculate correlations with fault conditions
    correlations = df_encoded[numerical_cols + [f'{fault_col}_encoded']].corr()[f'{fault_col}_encoded'].abs()
    correlations = correlations.sort_values(ascending=False)
    
    print(f"🔗 Top 10 Correlations with Fault Conditions:")
    print(correlations.head(10))
    
    # Strong correlations (>0.3)
    strong_correlations = correlations[correlations > 0.3]
    if len(strong_correlations) > 1:  # excluding self-correlation
        print(f"\n💪 Strong correlations (>0.3):")
        print(strong_correlations[:-1])  # exclude self-correlation
    else:
        print("\n📊 No strong correlations found (>0.3)")

# ============================================================================
# 7. BASIC VISUALIZATIONS
# ============================================================================

print(f"\n7️⃣ CREATING BASIC VISUALIZATIONS")
print("-" * 30)

# Create results directory if not exists
import os
results_dir = '../results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Plot 1: Fault Distribution
if fault_col:
    plt.figure(figsize=(10, 6))
    fault_counts.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Fault Condition Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Fault Condition', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{results_dir}/fault_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("📊 Fault distribution plot saved")

# Plot 2: Correlation Heatmap
if len(numerical_cols) > 1:
    plt.figure(figsize=(12, 10))
    correlation_matrix = df[numerical_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Correlation Matrix - Numerical Features', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{results_dir}/correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("📊 Correlation matrix saved")

# ============================================================================
# 8. SUMMARY REPORT
# ============================================================================

print(f"\n8️⃣ SUMMARY REPORT")
print("=" * 60)

print(f"📋 DATA QUALITY ASSESSMENT:")
print(f"   ✅ Total Records: {len(df):,}")
print(f"   ✅ Total Features: {len(df.columns)}")
print(f"   ✅ Numerical Features: {len(numerical_cols)}")
print(f"   ✅ Categorical Features: {len(categorical_cols)}")
print(f"   ✅ Missing Values: {total_missing}")

if fault_col:
    print(f"   ✅ Fault Classes: {len(df[fault_col].unique())}")
    print(f"   ✅ Class Balance: {normal_percent:.1f}% Normal")

print(f"\n🎯 KEY FINDINGS:")
print(f"   • Dataset successfully loaded from Excel")
print(f"   • {len(df):,} equipment records analyzed")
print(f"   • {len(df[fault_col].unique()) if fault_col else 'Unknown'} different fault conditions identified")
print(f"   • Data quality: {'Good' if total_missing == 0 else 'Needs cleaning'}")

print(f"\n🔄 NEXT STEPS:")
print(f"   1. Run Step 2: Data Cleaning & Feature Engineering")
print(f"   2. Handle missing values (if any)")
print(f"   3. Create derived features")
print(f"   4. Prepare data for modeling")

print(f"\n✅ STEP 1 COMPLETED SUCCESSFULLY!")
print(f"📁 Results saved in: {results_dir}/")
print("=" * 60)