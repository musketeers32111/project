import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# Load dataset
df = pd.read_csv('D:\\mlops project\\project\\data\\dummy_sensor_data.csv')

# Parse timestamps
# Convert 'Timestamp' to datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Extracting additional features from 'Timestamp' (if needed)
df['Hour'] = df['Timestamp'].dt.hour
# You can add more features like day of week, month, etc.

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Apply Label Encoding to 'Machine_ID' and 'Sensor_ID'
df['Machine_ID'] = label_encoder.fit_transform(df['Machine_ID'])
df['Sensor_ID'] = label_encoder.fit_transform(df['Sensor_ID'])

# If 'Reading' is the target variable
X = df.drop('Reading', axis=1)  # Features
y = df['Reading']  # Target

# Initialize StandardScaler
scaler = StandardScaler()

# Select numeric columns to scale
numeric_cols = ['Hour']  # Add other numeric columns if present

# Apply scaling to numeric columns
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

combined_df = pd.concat([X, y.reset_index(drop=True)], axis=1)

combined_df.to_csv('D:\\mlops project\\project\\data\\preprocessed_data.csv', index=False)
# # Split data into training and remaining data
X_train, X_remaining, y_train, y_remaining = train_test_split(X, y, test_size=0.3, random_state=42)

# # Split remaining data equally into validation and test set
X_val, X_test, y_val, y_test = train_test_split(X_remaining, y_remaining, test_size=0.5, random_state=42)

# Start an MLflow run
with mlflow.start_run():

    # Initialize the XGBoost model
    model = xgb.XGBRegressor(objective='reg:squarederror')

    # Train the model
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    # Log parameters and results
    mlflow.log_params(model.get_params())
    mlflow.log_metric('mse', mse)

    # Log the model
    mlflow.xgboost.log_model(model, "model")


param_grid = {
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    # Add other parameters here
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Log best parameters and model
mlflow.log_params(best_params)
mlflow.xgboost.log_model(best_model, "best_model")

model_name = "PredictiveMaintenanceModel"
mlflow.register_model(model_uri=f"runs:/{mlflow.active_run().info.run_id}/best_model", name=model_name)

# Assuming 'live_data' is your new dataset
# live_data = ...

live_data_processed = ...  # Apply necessary preprocessing

# Load the model from MLflow
model_path = f"models:/{model_name}/Production"
model = mlflow.pyfunc.load_model(model_path)

# Predict on new data
live_predictions = model.predict(live_data_processed)