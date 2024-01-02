import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mlflow_tracking import start_mlflow_run, log_params, log_metrics, log_model
import mlflow
import mlflow.sklearn

# Function to preprocess data
def preprocess_data(file_path, export_csv=False):
    # Load data
    df = pd.read_csv(file_path)

    # Convert Timestamp to datetime and extract relevant parts
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Hour'] = df['Timestamp'].dt.hour
    df['Day'] = df['Timestamp'].dt.day
    df['Month'] = df['Timestamp'].dt.month

    # One-hot encoding for categorical variables
    df_encoded = pd.get_dummies(df, columns=['Machine_ID', 'Sensor_ID'])

    # Drop columns not used in training
    X = df_encoded.drop(['Reading', 'Timestamp'], axis=1)
    y = df_encoded['Reading']

    if export_csv:
        # Export preprocessed data to CSV
        df_encoded.to_csv('D:\\mlops project\\project\\data\\preprocessed_data.csv', index=False)

    return X, y

# Function to train model
def train_model(X, y):
    with mlflow.start_run():
        # Splitting the dataset
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

        # Feature scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Train a Random Forest model
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train_scaled, y_train)

        # Log parameters, metrics, and model
        mlflow.log_params({"model_type": "RandomForestRegressor", "random_state": 42})
        mlflow.log_metric("mse", mean_squared_error(y_val, model.predict(X_val_scaled)))
        mlflow.sklearn.log_model(model, "model")

        return model, scaler

def load_model_and_scaler(model_path, scaler_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

# Function to make predictions and evaluate the model
def make_predictions_and_evaluate(model, scaler, X_data, y_true=None):
    # Scale the features
    X_scaled = scaler.transform(X_data)

    # Make predictions
    predictions = model.predict(X_scaled)
    print("Predictions:", predictions)

    # Evaluate the model if true labels are provided
    if y_true is not None:
        mse = mean_squared_error(y_true, predictions)
        print("Mean Squared Error:", mse)
        mse = mean_squared_error(y_true, predictions)
        rmse = mean_squared_error(y_true, predictions, squared=False)
        mae = mean_absolute_error(y_true, predictions)
        r2 = r2_score(y_true, predictions)

        print(f"Mean Squared Error (MSE): {mse}")
        print(f"Root Mean Squared Error (RMSE): {rmse}")
        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"R-squared: {r2}")



        


if __name__ == "__main__":
    # File path to your CSV data
    file_path = 'D:\\mlops project\\project\\data\\dummy_sensor_data.csv'

    # Preprocess the data and export to CSV
    X, y = preprocess_data(file_path, export_csv=True)

    # Train the model
    model, scaler = train_model(X, y)

    # Save the model and scaler
    model_path = 'random_forest_model.pkl'
    scaler_path = 'scaler.pkl'
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    # Load the model and scaler
    loaded_model, loaded_scaler = load_model_and_scaler(model_path, scaler_path)

    # If you have sample data for predictions and evaluation
    # X_sample, y_sample = [your sample data]
    # make_predictions_and_evaluate(loaded_model, loaded_scaler, X_sample, y_sample)

    # For demonstration, using the same data
    make_predictions_and_evaluate(loaded_model, loaded_scaler, X, y)
    
