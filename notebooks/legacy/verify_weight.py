import pandas as pd
import numpy as np

# Load data
train_path = r'c:\Storage\Smartapps\ABB Use case\Big_mart_sales_prediction\dataset\raw\train_v9rqX0R.csv'
test_path = r'c:\Storage\Smartapps\ABB Use case\Big_mart_sales_prediction\dataset\raw\test_AbJTz2l.csv'

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Combine to check across all data
combined = pd.concat([train_df, test_df], ignore_index=True)

print("Total rows:", len(combined))
print("Missing Item_Weight:", combined['Item_Weight'].isnull().sum())

# Check variance of Item_Weight for each Item_Identifier
# Filter out groups where all weights are missing or only 1 record exists
weight_stats = combined.groupby('Item_Identifier')['Item_Weight'].agg(['mean', 'std', 'count', 'nunique'])

# Items with variation in weight (std > 0)
varying_weights = weight_stats[weight_stats['std'] > 0]
print(f"\nNumber of items with varying weights: {len(varying_weights)}")

if len(varying_weights) > 0:
    print("Examples of items with varying weights:")
    print(varying_weights.head())
else:
    print("No items have varying weights (excluding floating point error).")

# Check if Item_Identifier is a good imputer
# How many missing weights can be filled by Item_Identifier?
# Items that have at least one non-missing weight across the dataset
items_with_weight = combined.dropna(subset=['Item_Weight'])['Item_Identifier'].unique()
missing_indices = combined[combined['Item_Weight'].isnull()].index
fillable_count = combined.loc[missing_indices, 'Item_Identifier'].isin(items_with_weight).sum()

print(f"\nTotal missing Item_Weight: {len(missing_indices)}")
print(f"Missing weights fillable by Item_Identifier mapping: {fillable_count}")
print(f"Remaining missing after Item_Identifier imputation: {len(missing_indices) - fillable_count}")

# Examples of best combination
print("\nExample Data Table for User (Item_Identifier vs Item_Weight):")
# Pick an item that has both missing and non-missing values
for item in items_with_weight:
    subset = combined[combined['Item_Identifier'] == item]
    if subset['Item_Weight'].isnull().any():
        print(f"Item: {item}")
        print(subset[['Item_Identifier', 'Outlet_Identifier', 'Item_Weight']])
        break
