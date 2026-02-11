import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Define paths
train_path = r'c:\Storage\Smartapps\ABB Use case\Big_mart_sales_prediction\dataset\raw\train_v9rqX0R.csv'
test_path = r'c:\Storage\Smartapps\ABB Use case\Big_mart_sales_prediction\dataset\raw\test_AbJTz2l.csv'
processed_dir = r'c:\Storage\Smartapps\ABB Use case\Big_mart_sales_prediction\dataset\processed'

# Create processed directory
os.makedirs(processed_dir, exist_ok=True)

# Load datasets
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
combined = pd.concat([train_df, test_df], ignore_index=True)

print("Loaded Train Shape:", train_df.shape)
print("Loaded Test Shape:", test_df.shape)

# Handle Missing Values - Item_Weight
print("Creating Item_Weight median mapping...")
item_weight_median = combined.groupby('Item_Identifier')['Item_Weight'].median()

# Impute missing Item_Weight
def impute_weight(row, median_map):
    if pd.isnull(row['Item_Weight']):
        return median_map.get(row['Item_Identifier'], np.nan)
    return row['Item_Weight']

print("Imputing missing Item_Weight...")
train_df['Item_Weight'] = train_df.apply(lambda x: impute_weight(x, item_weight_median), axis=1)
test_df['Item_Weight'] = test_df.apply(lambda x: impute_weight(x, item_weight_median), axis=1)
# Fallback
global_median = combined['Item_Weight'].median()
train_df['Item_Weight'].fillna(global_median, inplace=True)
test_df['Item_Weight'].fillna(global_median, inplace=True)


# Handle Missing Values - Outlet_Size (RF Imputation)
print("Imputing Outlet_Size using Random Forest...")
impute_df = pd.concat([train_df, test_df], ignore_index=True)
features = ['Outlet_Type', 'Outlet_Location_Type', 'Outlet_Establishment_Year']
target = 'Outlet_Size'

# Encode features
le = LabelEncoder()
impute_df_encoded = impute_df.copy()
for col in features:
    impute_df_encoded[col] = le.fit_transform(impute_df_encoded[col].astype(str))

# Split into known and unknown
known_size = impute_df_encoded[impute_df_encoded[target].notnull()]
unknown_size = impute_df_encoded[impute_df_encoded[target].isnull()]

if len(unknown_size) > 0:
    rf_imputer = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_imputer.fit(known_size[features], known_size[target])
    predicted_sizes = rf_imputer.predict(unknown_size[features])
    
    # Fill back
    impute_df.loc[impute_df[target].isnull(), target] = predicted_sizes
    
    # Split back to train/test
    train_df['Outlet_Size'] = impute_df.loc[:len(train_df)-1, 'Outlet_Size']
    test_df['Outlet_Size'] = impute_df.loc[len(train_df):, 'Outlet_Size'].values
else:
    print("No missing Outlet_Size found (strange).")

print("Missing Outlet_Size after imputation:", train_df['Outlet_Size'].isnull().sum())

# Clean Categorical Inconsistencies
mapping = {'LF': 'Low Fat', 'low fat': 'Low Fat', 'reg': 'Regular'}
train_df['Item_Fat_Content'] = train_df['Item_Fat_Content'].replace(mapping)
test_df['Item_Fat_Content'] = test_df['Item_Fat_Content'].replace(mapping)

print("Standardized Item_Fat_Content Categories:", train_df['Item_Fat_Content'].unique())

# Save Processed Data
print("Saving processed files...")
train_df.to_csv(os.path.join(processed_dir, 'cleaned_train.csv'), index=False)
test_df.to_csv(os.path.join(processed_dir, 'cleaned_test.csv'), index=False)

print(f"Files saved to {processed_dir}")
