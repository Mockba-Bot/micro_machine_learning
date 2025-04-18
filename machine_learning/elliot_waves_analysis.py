import os
import sys
import joblib
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from dotenv import load_dotenv
from historical_data import get_historical_data_limit
import warnings
warnings.filterwarnings("ignore")

# Add the directory containing your modules to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from machine_learning.bucket import download_model, upload_model

# Load environment variables from the .env file
dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env.micro.machine.learning'))
load_dotenv(dotenv_path=dotenv_path)

# Access the environment variables
CPU_COUNT = os.getenv("CPU_COUNT")
cpu_count = os.cpu_count() - int(CPU_COUNT)
BUCKET_NAME = os.getenv("BUCKET_NAME")  # Your bucket name

# Adjust limit based on interval
def get_limit_for_interval(interval):
    return {
        '1h': 500,
        '4h': 200,
        '1d': 100
    }.get(interval, 500)  # Default to 500 if interval is unknown

# Create feature dataset for training
def create_rf_dataset(data, look_back=60):
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back])
        Y.append(data[i + look_back])
    return np.array(X), np.array(Y)

# Train or update XGBoost model
def train_or_update_xgboost_model(symbol, interval, look_back=60):
    MODEL_KEY_XGBOOST = f'Mockba/elliot_waves_trained_models/{symbol}_{interval}_elliot_waves_model.joblib'
    model_path = f'temp/{symbol}_{interval}_elliot_waves_model.joblib'

    # Ensure the directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Get historical data
    limit = get_limit_for_interval(interval)
    data = get_historical_data_limit(symbol, interval, limit)

    # Feature extraction and scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['close'].values.reshape(-1, 1))

    # Prepare dataset for training
    X, Y = create_rf_dataset(scaled_data, look_back)
    X = X.reshape(X.shape[0], -1)  # Flatten input for XGBoost

    if download_model(BUCKET_NAME, MODEL_KEY_XGBOOST, model_path):
        best_rf = joblib.load(model_path)
        print(f"Updating existing XGBoost model for {symbol} on {interval} interval.")
        best_rf.fit(X, Y, xgb_model=best_rf.get_booster())  # Update model with new data
    else:
        print(f"Training new XGBoost model for {symbol} on {interval} interval.")
        
        param_dist = {
            'n_estimators': randint(100, 800),
            'learning_rate': [0.01, 0.05, 0.1], 
            'max_depth': [3, 5, 7, 10],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.3, 0.5, 0.7, 1.0],
            'gamma': [0, 0.1, 0.2, 0.3],
            'min_child_weight': [1, 3, 5, 7],
            'reg_alpha': [0, 0.01, 0.1, 1],
            'reg_lambda': [1, 1.5, 2, 3]
        }

        # Hyperparameter tuning
        xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
        xgb_random = RandomizedSearchCV(
            estimator=xgb_reg, 
            param_distributions=param_dist, 
            n_iter=200, 
            cv=3, 
            verbose=0, 
            random_state=42, 
            n_jobs=cpu_count
        )
        xgb_random.fit(X, Y)

        # Select best model
        best_rf = xgb_random.best_estimator_

    # Save the updated model
    joblib.dump(best_rf, model_path, compress=3)

    upload_model(BUCKET_NAME, MODEL_KEY_XGBOOST, model_path)

    # Delete the local file after uploading
    if os.path.exists(model_path):
        os.remove(model_path)
    else:
        print(f"Local file {model_path} does not exist.")

# Main function to train or update models for multiple intervals
def train_models(symbol, intervals):
    for interval in intervals:
        try:
            train_or_update_xgboost_model(symbol, interval)
        except Exception as e:
            print(f"Skipping {symbol}/{interval} due to error: {str(e)}")
            continue

# Run training for selected intervals
# if __name__ == "__main__":
#     symbol = "PERP_APT_USDC"
#     intervals = ["1h", "4h", "1d"]
#     train_models(symbol, intervals)
