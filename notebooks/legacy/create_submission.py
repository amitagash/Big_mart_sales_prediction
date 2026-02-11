import pandas as pd
import numpy as np
import os
from xgboost import XGBRegressor

# Define paths
processed_dir = r'c:\Storage\Smartapps\ABB Use case\Big_mart_sales_prediction\dataset\processed'
train_path = os.path.join(processed_dir, 'feat_eng_train.csv')
test_path = os.path.join(processed_dir, 'feat_eng_test.csv')
submission_path = r'c:\Storage\Smartapps\ABB Use case\Big_mart_sales_prediction\model_training\submission_v2.csv'

# Load data
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Prepare Data
cols_to_drop = ['Item_Identifier', 'Outlet_Identifier', 'Item_Type', 'Outlet_Establishment_Year']

X = train_df.drop(columns=['Item_Outlet_Sales'])
y = train_df['Item_Outlet_Sales']

X = X.drop(columns=cols_to_drop, errors='ignore')
test_X = test_df.drop(columns=cols_to_drop, errors='ignore')

# Ensure columns match (e.g. if one-hot encoding missed categories in test or train)
# Align columns: Test must have same columns as Train (and in same order)
# Add missing cols to test as 0
missing_cols = set(X.columns) - set(test_X.columns)
for c in missing_cols:
    test_X[c] = 0
# Drop extra cols in test if any (shouldn't be, but good practice)
extra_cols = set(test_X.columns) - set(X.columns)
test_X = test_X.drop(columns=extra_cols, errors='ignore')
# Reorder
test_X = test_X[X.columns]

print("Training Data Shape:", X.shape)
print("Test Data Shape:", test_X.shape)

# Train XGBoost
print("Training XGBoost...")
xgb = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
xgb.fit(X, y)

# Predict
print("Predicting...")
predictions = xgb.predict(test_X)

# Create Submission
submission = pd.DataFrame({
    'Item_Identifier': test_df['Item_Identifier'],
    'Outlet_Identifier': test_df['Outlet_Identifier'],
    'Item_Outlet_Sales': predictions
})

# Handle negative predictions if any (Sales can't be negative)
submission['Item_Outlet_Sales'] = submission['Item_Outlet_Sales'].apply(lambda x: 0 if x < 0 else x)

submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
print(submission.head())
