import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

# Define paths
processed_dir = r'c:\Storage\Smartapps\ABB Use case\Big_mart_sales_prediction\dataset\processed'
train_path = os.path.join(processed_dir, 'cleaned_train.csv')
test_path = os.path.join(processed_dir, 'cleaned_test.csv')

# Load cleaned datasets
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print("Loaded Cleaned Train Shape:", train_df.shape)

# Combine train and test for consistent feature engineering
train_len = len(train_df)
data_df = pd.concat([train_df, test_df], ignore_index=True)

# Feature Engineering: Item_Visibility_MeanRatio
# We observed in EDA that visibility can be 0, which is likely missing data.
# We'll replace 0 visibility with the mean visibility of that product.
visibility_avg = data_df.pivot_table(values='Item_Visibility', index='Item_Identifier')

def impute_visibility_mean(cols):
    visibility = cols[0]
    item = cols[1]
    if visibility == 0:
        return visibility_avg['Item_Visibility'][visibility_avg.index == item].values[0]
    return visibility

print("Imputing 0 visibility with mean...")
data_df['Item_Visibility'] = data_df[['Item_Visibility', 'Item_Identifier']].apply(impute_visibility_mean, axis=1)

# Now calculate the ratio
print("Creating Item_Visibility_MeanRatio...")
visibility_avg = data_df.pivot_table(values='Item_Visibility', index='Item_Identifier') # Recalculate after imputation
data_df['Item_Visibility_MeanRatio'] = data_df.apply(lambda x: x['Item_Visibility'] / visibility_avg['Item_Visibility'][visibility_avg.index == x['Item_Identifier']].values[0], axis=1)

# Create Item_Visibility_Ratio_OutletSize
print("Creating Item_Visibility_Ratio_OutletSize...")
# Calculate mean visibility by Outlet_Size
visibility_size_avg = data_df.pivot_table(values='Item_Visibility', index='Outlet_Size')

def get_visibility_size_ratio(row):
    size = row['Outlet_Size']
    visibility = row['Item_Visibility']
    if size not in visibility_size_avg.index:
        return 1.0 # Fallback if size is somehow missing or new
    mean_vis = visibility_size_avg.loc[size, 'Item_Visibility']
    if mean_vis == 0:
        return 0 # Avoid division by zero
    return visibility / mean_vis

data_df['Item_Visibility_Ratio_OutletSize'] = data_df.apply(get_visibility_size_ratio, axis=1)

# Split back to train and test
train_df = data_df[:train_len].copy()
test_df = data_df[train_len:].copy()

# Feature Generation: Outlet_Years
train_df['Outlet_Years'] = 2013 - train_df['Outlet_Establishment_Year']
test_df['Outlet_Years'] = 2013 - test_df['Outlet_Establishment_Year']

# Feature Generation: Item_Type_Combined
train_df['Item_Type_Combined'] = train_df['Item_Identifier'].apply(lambda x: x[0:2])
train_df['Item_Type_Combined'] = train_df['Item_Type_Combined'].map({'FD': 'Food', 'NC': 'Non-Consumable', 'DR': 'Drinks'})

test_df['Item_Type_Combined'] = test_df['Item_Identifier'].apply(lambda x: x[0:2])
test_df['Item_Type_Combined'] = test_df['Item_Type_Combined'].map({'FD': 'Food', 'NC': 'Non-Consumable', 'DR': 'Drinks'})

# Categorical Encoding
# Manual Ordinal Encoding
# Outlet_Size: Small < Medium < High
size_mapping = {'Small': 0, 'Medium': 1, 'High': 2}
train_df['Outlet_Size'] = train_df['Outlet_Size'].map(size_mapping)
test_df['Outlet_Size'] = test_df['Outlet_Size'].map(size_mapping)

# Outlet_Location_Type: Tier 1 < Tier 2 < Tier 3
loc_mapping = {'Tier 1': 0, 'Tier 2': 1, 'Tier 3': 2}
train_df['Outlet_Location_Type'] = train_df['Outlet_Location_Type'].map(loc_mapping)
test_df['Outlet_Location_Type'] = test_df['Outlet_Location_Type'].map(loc_mapping)

print("Manual Encoding completed for Outlet_Size and Outlet_Location_Type.")

# One-Hot Encoding for Nominal Categories
train_df = pd.get_dummies(train_df, columns=['Item_Fat_Content', 'Outlet_Type', 'Item_Type_Combined'])
test_df = pd.get_dummies(test_df, columns=['Item_Fat_Content', 'Outlet_Type', 'Item_Type_Combined'])

print("Train Shape after encoding:", train_df.shape)
print("Test Shape after encoding:", test_df.shape)

# Drop Correlated/Redundant Features
# Outlet_Establishment_Year is perfectly correlated with Outlet_Years (Years = 2013 - Est_Year)
train_df = train_df.drop(columns=['Outlet_Establishment_Year'])
test_df = test_df.drop(columns=['Outlet_Establishment_Year'])
print("Dropped Outlet_Establishment_Year.")

# Save Feature Engineered Data (keeping IDs for now, dropping only strictly unnecessary for calculations if any)
# Actually, let's keep everything and let model training select columns or drop IDs there.
# But `Item_Identifier` and `Outlet_Identifier` are non-numeric strings, so we can't feed them to most models directly.
# We will drop them in the training script or just leave them and exclude them from X features.

train_df.to_csv(os.path.join(processed_dir, 'feat_eng_train.csv'), index=False)
test_df.to_csv(os.path.join(processed_dir, 'feat_eng_test.csv'), index=False)

print(f"Feature engineered files saved to {processed_dir}")
