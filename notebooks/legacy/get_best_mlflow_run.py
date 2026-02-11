import mlflow
import pandas as pd

# Set tracking URI if it was set in the tuning script (defaults to ./mlruns)
# mlflow.set_tracking_uri("http://localhost:5000") 

try:
    experiment = mlflow.get_experiment_by_name("BigMart_Sales_Optimization")
    if experiment is None:
        print("Experiment not found.")
    else:
        experiment_id = experiment.experiment_id
        
        # Search runs
        runs = mlflow.search_runs(experiment_ids=[experiment_id], order_by=["metrics.cv_rmse ASC"])
        
        if not runs.empty:
            best_run = runs.iloc[0]
            print("Best Run Found:")
            print(f"  Run ID: {best_run.run_id}")
            print(f"  CV RMSE: {best_run['metrics.cv_rmse']}")
            print("  Parameters:")
            # Filter for params
            params = {k.replace("params.", ""): v for k, v in best_run.items() if k.startswith("params.")}
            for k, v in params.items():
                print(f"    {k}: {v}")
                
            # Save params to file
            import json
            with open("best_params.json", "w") as f:
                json.dump(params, f, indent=4)
            print("Saved best_params.json")
        else:
            print("No runs found.")

except Exception as e:
    print(f"Error accessing MLflow: {e}")
