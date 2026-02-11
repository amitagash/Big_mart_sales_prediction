import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Define paths
processed_dir = r'c:\Storage\Smartapps\ABB Use case\Big_mart_sales_prediction\dataset\processed'
train_path = os.path.join(processed_dir, 'feat_eng_train.csv')
test_path = os.path.join(processed_dir, 'feat_eng_test.csv')

# Load data
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print("Loaded Train Shape:", train_df.shape)

# Prepare Data
X = train_df.drop(columns=['Item_Outlet_Sales'])
y = train_df['Item_Outlet_Sales']

cols_to_drop = ['Item_Identifier', 'Outlet_Identifier', 'Item_Type']
X = X.drop(columns=cols_to_drop, errors='ignore')

if 'Outlet_Establishment_Year' in X.columns:
    X = X.drop(columns=['Outlet_Establishment_Year'])

# Split Data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate_model(name, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print(f"{name} RMSE: {rmse}")
    return rmse

# Train Models
print("\n--- Model Evaluation ---")
evaluate_model("Linear Regression", LinearRegression())
evaluate_model("Decision Tree", DecisionTreeRegressor(random_state=42))
evaluate_model("Random Forest", RandomForestRegressor(random_state=42, n_estimators=100))
evaluate_model("XGBoost", XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1))
