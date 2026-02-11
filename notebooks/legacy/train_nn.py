import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define paths
processed_dir = r'c:\Storage\Smartapps\ABB Use case\Big_mart_sales_prediction\dataset\processed'
train_path = os.path.join(processed_dir, 'feat_eng_train.csv')
test_path = os.path.join(processed_dir, 'feat_eng_test.csv')

# Load data
print("Loading data...")
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Prepare Data
X = train_df.drop(columns=['Item_Outlet_Sales'])
y = train_df['Item_Outlet_Sales']

# Drop non-numeric identifier columns
cols_to_drop = ['Item_Identifier', 'Outlet_Identifier', 'Item_Type']
X = X.drop(columns=cols_to_drop, errors='ignore')

if 'Outlet_Establishment_Year' in X.columns:
    X = X.drop(columns=['Outlet_Establishment_Year'])
    
print(f"Input Shape: {X.shape}")

# Split Data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling (Crucial for NN)
scaler_X = RobustScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)

# Scale Target as well
from sklearn.preprocessing import StandardScaler
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_val_scaled = scaler_y.transform(y_val.values.reshape(-1, 1))

# Define Model with Regularization
from tensorflow.keras import regularizers

model = keras.Sequential([
    layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.4),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(1) # Output layer
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001), 
    loss='mean_squared_error',
    metrics=['root_mean_squared_error']
)

model.summary()

# Train Model
print("Training Neural Network...")
history = model.fit(
    X_train_scaled, y_train_scaled,
    validation_data=(X_val_scaled, y_val_scaled),
    batch_size=32,
    epochs=150,
    verbose=1,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
    ]
)

# Evaluate
print("\n--- Evaluation ---")
y_pred_scaled = model.predict(X_val_scaled)
# Inverse transform predictions
y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"Neural Network RMSE (Refined): {rmse}")

with open(os.path.join(processed_dir, 'nn_rmse_refined.txt'), 'w') as f:
    f.write(str(rmse))
    
# Plot Loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss Progress (Scaled Target)')
plt.ylabel('Loss (MSE)')
plt.xlabel('Epoch')
plt.legend()
plt.savefig(os.path.join(processed_dir, 'nn_training_loss.png'))
print(f"Loss plot saved to {processed_dir}\\nn_training_loss.png")
