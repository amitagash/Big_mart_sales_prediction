import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

# Load Data
processed_dir = r'c:\Storage\Smartapps\ABB Use case\Big_mart_sales_prediction\dataset\processed'
train_path = os.path.join(processed_dir, 'feat_eng_train.csv')

if not os.path.exists(train_path):
    print("Feature engineered data not found. Run feature_engineering.py first.")
    exit()

df = pd.read_csv(train_path)

# Drop non-numeric
numeric_df = df.select_dtypes(include=[np.number])

# Calculate Correlation
corr_matrix = numeric_df.corr()

# Print High Correlations
print("--- High Correlations (> 0.8) ---")
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > 0.8:
            colname_i = corr_matrix.columns[i]
            colname_j = corr_matrix.columns[j]
            print(f"{colname_i} vs {colname_j}: {corr_matrix.iloc[i, j]:.2f}")
            high_corr_pairs.append((colname_i, colname_j))

if not high_corr_pairs:
    print("No highly correlated features found > 0.8")

# Saving Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig(os.path.join(processed_dir, 'correlation_heatmap.png'))
print(f"Correlation heatmap saved to {processed_dir}\\correlation_heatmap.png")
