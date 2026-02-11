import pandas as pd
import numpy as np
import optuna
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import os

# Set MLflow tracking URI (optional, defaults to ./mlruns)
# mlflow.set_tracking_uri("http://localhost:5000") 
mlflow.set_experiment("BigMart_Sales_Optimization")

# Load Data
processed_dir = r'c:\Storage\Smartapps\ABB Use case\Big_mart_sales_prediction\dataset\processed'
train_path = os.path.join(processed_dir, 'feat_eng_train.csv')
train_df = pd.read_csv(train_path)

# Separate features and target
X = train_df.drop('Item_Outlet_Sales', axis=1)
y = train_df['Item_Outlet_Sales']

# Drop non-numeric identifier columns
cols_to_drop = ['Item_Identifier', 'Outlet_Identifier', 'Item_Type']
X = X.drop(columns=cols_to_drop, errors='ignore')
# Also drop Outlet_Establishment_Year if present, as instructed in feature engineering
if 'Outlet_Establishment_Year' in X.columns:
    X = X.drop(columns=['Outlet_Establishment_Year'])

print("Training Features:", X.columns.tolist())

def objective(trial):
    with mlflow.start_run(nested=True):
        # 1. Suggest Scaler
        scaler_name = trial.suggest_categorical("scaler", ["Standard", "MinMax", "Robust"])
        if scaler_name == "Standard":
            scaler = StandardScaler()
        elif scaler_name == "MinMax":
            scaler = MinMaxScaler()
        else:
            scaler = RobustScaler()
            
        # 2. Suggest XGBoost Hyperparameters
        param = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": 42,
            "n_jobs": -1
        }
        
        # 3. Create Pipeline
        model = XGBRegressor(**param)
        pipeline = Pipeline([
            ("scaler", scaler),
            ("model", model)
        ])
        
        # 4. Cross-Validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        # XGBoost minimizes MSE, but we want RMSE. cross_val_score returns neg_mean_squared_error
        scores = cross_val_score(pipeline, X, y, cv=kf, scoring="neg_root_mean_squared_error")
        rmse = -scores.mean()
        
        # 5. Log to MLflow
        mlflow.log_params(param)
        mlflow.log_param("scaler", scaler_name)
        mlflow.log_metric("cv_rmse", rmse)
        
        return rmse

if __name__ == "__main__":
    print("Starting Optuna optimization...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20) # Running 20 trials for demonstration

    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        
    # Retrain best model on full data and log artifacts
    print("\nRetraining best model and logging artifacts...")
    with mlflow.start_run(run_name="Best_Model"):
        # Reconstruct best pipeline
        best_params = trial.params.copy()
        scaler_name = best_params.pop("scaler")
        
        if scaler_name == "Standard":
            scaler = StandardScaler()
        elif scaler_name == "MinMax":
            scaler = MinMaxScaler()
        else:
            scaler = RobustScaler()
            
        model = XGBRegressor(**best_params, random_state=42, n_jobs=-1)
        pipeline = Pipeline([
            ("scaler", scaler),
            ("model", model)
        ])
        
        pipeline.fit(X, y)
        
        # Log params and metrics
        mlflow.log_params(best_params)
        mlflow.log_param("scaler", scaler_name)
        mlflow.log_metric("final_rmse", trial.value)
        
        # Log model
        mlflow.sklearn.log_model(pipeline, "model")
        
        # Create and log plots
        y_pred = pipeline.predict(X)
        
        # Actual vs Predicted Plot
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y, y=y_pred, alpha=0.5)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
        plt.xlabel("Actual Sales")
        plt.ylabel("Predicted Sales")
        plt.title(f"Actual vs Predicted (RMSE: {trial.value:.2f})")
        plt.savefig("actual_vs_predicted.png")
        mlflow.log_artifact("actual_vs_predicted.png")
        plt.close()
        
        # Residual Plot
        residuals = y - y_pred
        plt.figure(figsize=(10, 6))
        sns.distplot(residuals)
        plt.title("Residuals Distribution")
        plt.xlabel("Residual")
        plt.savefig("residuals.png")
        mlflow.log_artifact("residuals.png")
        plt.close()
        
        print("Optimization complete. Check MLflow for details.")
