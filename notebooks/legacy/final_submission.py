import pandas as pd
import numpy as np
import os
import json
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline

# Define paths
processed_dir = r'c:\Storage\Smartapps\ABB Use case\Big_mart_sales_prediction\dataset\processed'
train_path = os.path.join(processed_dir, 'feat_eng_train.csv')
test_path = os.path.join(processed_dir, 'feat_eng_test.csv')
# Switch to GridSearch params as they gave better CV score (1081 vs 1099)
params_path = r'c:\Storage\Smartapps\ABB Use case\Big_mart_sales_prediction\model_training\best_grid_params.json'
submission_path = r'c:\Storage\Smartapps\ABB Use case\Big_mart_sales_prediction\model_training\submission_final_grid.csv'

# Load data
print("Loading data...")
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Prepare Data
X_train = train_df.drop(columns=['Item_Outlet_Sales'])
y_train = train_df['Item_Outlet_Sales']
X_test = test_df.copy() # Test data doesn't have Item_Outlet_Sales

# Drop non-numeric columns
cols_to_drop = ['Item_Identifier', 'Outlet_Identifier', 'Item_Type']
X_train = X_train.drop(columns=cols_to_drop, errors='ignore')
X_test = X_test.drop(columns=cols_to_drop, errors='ignore')

if 'Outlet_Establishment_Year' in X_train.columns:
    X_train = X_train.drop(columns=['Outlet_Establishment_Year'])
if 'Outlet_Establishment_Year' in X_test.columns:
    X_test = X_test.drop(columns=['Outlet_Establishment_Year'])

# Load best parameters
with open(params_path, 'r') as f:
    best_params = json.load(f)

scaler_name = best_params.pop('scaler', 'Standard') # Default to Standard if missing
print(f"Using Scaler: {scaler_name}")
print("Best Params:", best_params)

# Select Scaler
if scaler_name == "Standard":
    scaler = StandardScaler()
elif scaler_name == "MinMax":
    scaler = MinMaxScaler()
else:
    scaler = RobustScaler()

# Train Model
print("Training final model...")
# Ensure numeric params are converted correctly if read from JSON as strings (though json.load handles numbers usually)
# Explicitly casting essential ones just in case
if 'n_estimators' in best_params: best_params['n_estimators'] = int(best_params['n_estimators'])
if 'max_depth' in best_params: best_params['max_depth'] = int(best_params['max_depth'])
if 'n_jobs' in best_params: best_params['n_jobs'] = int(best_params['n_jobs'])

model = XGBRegressor(**best_params)
pipeline = Pipeline([
    ("scaler", scaler),
    ("model", model)
])

pipeline.fit(X_train, y_train)

# Predict
print("Predicting on test set...")
predictions = pipeline.predict(X_test)
# Ensure no negative sales
predictions = np.maximum(predictions, 0)

# Create Submission
print(f"Saving submission to {submission_path}...")
submission = pd.read_csv(test_path)[['Item_Identifier', 'Outlet_Identifier']]
submission['Item_Outlet_Sales'] = predictions
submission.to_csv(submission_path, index=False)

print("Done!")
