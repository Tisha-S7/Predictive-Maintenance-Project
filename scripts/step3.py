# File: scripts/step3_eda_visualization.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# Set plotting parameters
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

print("="*60)
print("    🔧 PREDICTIVE MAINTENANCE - STEP 3: EDA & VISUALIZATION")
print("="*60)

# Load dataset
try:
    df = pd.read_excel('data/maintenance_data.xlsx')
    print("✅ Data loaded successfully.\n")
except FileNotFoundError:
    print("❌ Error: 'data/maintenance_data.xlsx' not found.")
    exit()

# Basic Info
print("🔹 Dataset Info:")
print(df.info())

print("\n🔹 First 5 Rows:")
print(df.head())

# Missing Values
print("\n🔹 Missing Values:")
print(df.isnull().sum())

# Descriptive statistics
print("\n🔹 Summary Statistics:")
print(df.describe())

# Encode categorical columns
cat_cols = df.select_dtypes(include='object').columns
if len(cat_cols) > 0:
    print(f"\n🔹 Encoding categorical columns: {list(cat_cols)}")
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

# Correlation Heatmap
print("\n📌 Generating Correlation Heatmap...")
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig('outputs/eda_correlation_heatmap.png')
plt.close()

# Histograms
print("📌 Plotting Histograms...")
df.hist(bins=30, figsize=(15, 10), edgecolor='black')
plt.suptitle("Feature Distributions")
plt.tight_layout()
plt.savefig('outputs/eda_histograms.png')
plt.close()

# Boxplots
print("📌 Plotting Boxplots...")
for col in df.select_dtypes(include=np.number).columns:
    print(f"   📦 {col}")
    try:
        plt.figure()
        sns.boxplot(x=df[col], color='skyblue')
        plt.title(f'Boxplot of {col}')
        plt.tight_layout()
        plt.savefig(f'outputs/boxplot_{col}.png')
        plt.close()
    except Exception as e:
        print(f"   ⚠️ Skipped {col} due to error: {e}")

# Interactive Scatter Matrix
print("📌 Creating Interactive Scatter Matrix...")
try:
    fig = px.scatter_matrix(df,
                            dimensions=df.select_dtypes(include=np.number).columns,
                            color=df.columns[-1])  # Assumes target is last
    fig.update_layout(title='Scatter Matrix of Features', dragmode='select', height=800)
    fig.write_html("outputs/interactive_scatter_matrix.html")
except Exception as e:
    print(f"   ⚠️ Could not create interactive scatter matrix: {e}")

# Optional Plotly bar chart
if 'MachineType' in df.columns and 'Failure' in df.columns:
    print("📌 Plotly Bar Chart of Failures by Machine Type...")
    try:
        failure_count = df.groupby('MachineType')['Failure'].sum().reset_index()
        fig = px.bar(failure_count, x='MachineType', y='Failure', color='Failure',
                     title='Failures by Machine Type')
        fig.write_html("outputs/failure_by_machine_type.html")
    except Exception as e:
        print(f"   ⚠️ Could not create bar chart: {e}")

print("\n✅ EDA & Visualization Complete. Check the /outputs folder.")
