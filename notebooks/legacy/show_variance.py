import pandas as pd
import numpy as np

# Load data
train_path = r'c:\Storage\Smartapps\ABB Use case\Big_mart_sales_prediction\dataset\raw\train_v9rqX0R.csv'
test_path = r'c:\Storage\Smartapps\ABB Use case\Big_mart_sales_prediction\dataset\raw\test_AbJTz2l.csv'

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
combined = pd.concat([train_df, test_df], ignore_index=True)

# 1. Check Variance
print("--- Variance Analysis ---")
weight_stats = combined.groupby('Item_Identifier')['Item_Weight'].std()
varying_items = weight_stats[weight_stats > 0]
print(f"Number of items with varying weights (std > 0): {len(varying_items)}")

if len(varying_items) > 0:
    print("\nItems with varying weights:")
    print(varying_items.head())
else:
    print("\nConfirmed: All items have consistent weights (variance is 0 or NaN if only 1 value).")

# 2. Plot Data Table for User
print("\n--- Sample Data Table (Item: FDA15) ---")
item_id = 'FDA15'
subset = combined[combined['Item_Identifier'] == item_id][['Item_Identifier', 'Outlet_Identifier', 'Item_Weight']]
print(subset)

# 3. Check another item with missing values
print("\n--- Sample Data Table with Missing Values (Item: FDY38) ---")
# Find an item with at least one missing weight and one present weight
mixed_items = combined.groupby('Item_Identifier')['Item_Weight'].apply(lambda x: x.isnull().any() and x.notnull().any())
mixed_item_id = mixed_items[mixed_items].index[0] if mixed_items.any() else None

if mixed_item_id:
    subset_mixed = combined[combined['Item_Identifier'] == mixed_item_id][['Item_Identifier', 'Outlet_Identifier', 'Item_Weight']]
    print(f"Item: {mixed_item_id}")
    print(subset_mixed)
else:
    print("No items found with mixed missing/present weights.")
