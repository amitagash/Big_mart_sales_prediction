import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import RobustScaler

# Set seeds
np.random.seed(42)
tf.random.set_seed(42)

# Define paths
processed_dir = r'c:\Storage\Smartapps\ABB Use case\Big_mart_sales_prediction\dataset\processed'
train_path = os.path.join(processed_dir, 'feat_eng_train.csv')
test_path = os.path.join(processed_dir, 'feat_eng_test.csv')
submission_path = r'c:\Storage\Smartapps\ABB Use case\Big_mart_sales_prediction\model_training\submission_nn.csv'

# Load data
print("Loading data...")
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Prepare Data
X = train_df.drop(columns=['Item_Outlet_Sales'])
y = train_df['Item_Outlet_Sales']
X_test = test_df.copy()

# Drop non-numeric identifier columns
cols_to_drop = ['Item_Identifier', 'Outlet_Identifier', 'Item_Type']
X = X.drop(columns=cols_to_drop, errors='ignore')
X_test = X_test.drop(columns=cols_to_drop, errors='ignore')

# Explicitly drop target from test if present (artifact of concat/split)
if 'Item_Outlet_Sales' in X_test.columns:
    X_test = X_test.drop(columns=['Item_Outlet_Sales'])

if 'Outlet_Establishment_Year' in X.columns:
    X = X.drop(columns=['Outlet_Establishment_Year'])
if 'Outlet_Establishment_Year' in X_test.columns:
    X_test = X_test.drop(columns=['Outlet_Establishment_Year'])
    
# Scaling
scaler_X = RobustScaler()
X_scaled = scaler_X.fit_transform(X)
X_test_scaled = scaler_X.transform(X_test)

# Scale Target
from sklearn.preprocessing import StandardScaler
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

# Define Model (Refined)
from tensorflow.keras import regularizers

model = keras.Sequential([
    layers.Dense(256, activation='relu', input_shape=(X.shape[1],), kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.4),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(1)
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001), 
    loss='mean_squared_error'
)

# Train on Full Data
print("Training Neural Network on full dataset...")
model.fit(
    X_scaled, y_scaled,
    batch_size=32,
    epochs=150,
    verbose=1,
    callbacks=[
        keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-5)
    ]
)

# Predict
print("Predicting on test set...")
predictions_scaled = model.predict(X_test_scaled)
predictions = scaler_y.inverse_transform(predictions_scaled).flatten()
predictions = np.maximum(predictions, 0) # No negative sales

# Create Submission
print(f"Saving submission to {submission_path}...")
submission = pd.read_csv(test_path)[['Item_Identifier', 'Outlet_Identifier']]
submission['Item_Outlet_Sales'] = predictions
submission.to_csv(submission_path, index=False)

print("Done!")
