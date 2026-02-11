import pandas as pd
import os

# Define paths
train_path = r'c:\Storage\Smartapps\ABB Use case\Big_mart_sales_prediction\dataset\raw\train_v9rqX0R.csv'
test_path = r'c:\Storage\Smartapps\ABB Use case\Big_mart_sales_prediction\dataset\raw\test_AbJTz2l.csv'

# Check if files exist
if not os.path.exists(train_path):
    print(f"Error: Train file not found at {train_path}")
if not os.path.exists(test_path):
    print(f"Error: Test file not found at {test_path}")

# Load datasets
try:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print("Train Shape:", train_df.shape)
    print("Test Shape:", test_df.shape)
    
    print("\nTrain Columns:\n", train_df.columns.tolist())
    print("\nTrain Info:")
    print(train_df.info())
    print("\nMissing Values in Train:\n", train_df.isnull().sum())
    
    print("\nTest Info:")
    print(test_df.info())
    print("\nMissing Values in Test:\n", test_df.isnull().sum())

    print("\nSample Data (Train head):")
    print(train_df.head())

except Exception as e:
    print(f"An error occurred: {e}")
