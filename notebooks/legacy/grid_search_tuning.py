import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import json
import os
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

# Set MLflow
mlflow.set_experiment("BigMart_Sales_GridSearch")

# Load Data
processed_dir = r'c:\Storage\Smartapps\ABB Use case\Big_mart_sales_prediction\dataset\processed'
train_path = os.path.join(processed_dir, 'feat_eng_train.csv')
train_df = pd.read_csv(train_path)

# Prepare Data
X = train_df.drop('Item_Outlet_Sales', axis=1)
y = train_df['Item_Outlet_Sales']

# Drop non-numeric columns
cols_to_drop = ['Item_Identifier', 'Outlet_Identifier', 'Item_Type']
X = X.drop(columns=cols_to_drop, errors='ignore')
if 'Outlet_Establishment_Year' in X.columns:
    X = X.drop(columns=['Outlet_Establishment_Year'])

# Load best params from Optuna to center the grid
params_path = r'c:\Storage\Smartapps\ABB Use case\Big_mart_sales_prediction\model_training\best_params.json'
try:
    with open(params_path, 'r') as f:
        best_optuna_params = json.load(f)
    print("Loaded Optuna params:", best_optuna_params)
    
    # Extract centers (casting to appropriate types)
    center_n_est = int(best_optuna_params.get('n_estimators', 100))
    center_depth = int(best_optuna_params.get('max_depth', 5))
    center_lr = float(best_optuna_params.get('learning_rate', 0.1))
    
    # Define Grid around these values
    param_grid = {
        'model__n_estimators': [center_n_est - 50, center_n_est, center_n_est + 50],
        'model__max_depth': [center_depth - 1, center_depth, center_depth + 1],
        'model__learning_rate': [center_lr * 0.9, center_lr, center_lr * 1.1],
        # Keep some fixed or narrow
        'model__subsample': [0.8, 0.9], 
        'model__colsample_bytree': [0.8, 0.9]
    }
    
    # Scaler selection
    scaler_type = best_optuna_params.get('scaler', 'Robust')
    if scaler_type == 'Standard':
        scaler = StandardScaler()
    elif scaler_type == 'MinMax':
        scaler = MinMaxScaler()
    else:
        scaler = RobustScaler()
        
    print(f"Using Scaler: {scaler_type}")
    print("Grid:", param_grid)

except Exception as e:
    print(f"Could not load best_params.json: {e}. Using default grid.")
    scaler = RobustScaler()
    param_grid = {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [3, 5, 7],
        'model__learning_rate': [0.01, 0.1, 0.2]
    }

# Ensure n_estimators > 0
param_grid['model__n_estimators'] = [x for x in param_grid['model__n_estimators'] if x > 0]

# Pipeline
pipeline = Pipeline([
    ('scaler', scaler),
    ('model', XGBRegressor(random_state=42, n_jobs=-1))
])

# GridSearchCV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='neg_root_mean_squared_error',
    cv=kf,
    n_jobs=-1,
    verbose=1
)

print("Starting GridSearchCV...")
with mlflow.start_run(run_name="GridSearch_Run"):
    grid_search.fit(X, y)
    
    best_rmse = -grid_search.best_score_
    best_params = grid_search.best_params_
    
    print(f"\nBest GridSearch RMSE: {best_rmse}")
    print("Best GridSearch Params:", best_params)
    
    # Log to MLflow
    mlflow.log_params(best_params)
    mlflow.log_metric("best_cv_rmse", best_rmse)
    mlflow.sklearn.log_model(grid_search.best_estimator_, "model")
    
    # Save best grid params
    clean_params = {k.replace('model__', ''): v for k, v in best_params.items()}
    clean_params['scaler'] = 'Robust' # Assuming we stuck with the best scaler from Optuna
    
    with open("best_grid_params.json", "w") as f:
        json.dump(clean_params, f, indent=4)
    print("Saved best_grid_params.json")
