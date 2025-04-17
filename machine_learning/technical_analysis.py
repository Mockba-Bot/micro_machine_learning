import sys
import os
import joblib
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import randint
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from historical_data import get_historical_data_limit
import requests
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
cpu_count = os.cpu_count()-int(CPU_COUNT)
BUCKET_NAME = os.getenv("BUCKET_NAME")  # Your bucket name


############################################################################################################
# Linear Regression Model
############################################################################################################

def add_features(df):
    """
    Generate additional features for the model.
    
    Parameters:
    df (DataFrame): The input dataframe with 'close' and 'volume' columns.

    Returns:
    DataFrame: The dataframe with new feature columns.
    """
    df = df.copy()
    
    df['start_timestamp'] = pd.to_datetime(df['start_timestamp'])  # Ensure it's datetime format
    df['hour'] = df['start_timestamp'].dt.hour  # Extract hour
    df['day_of_week'] = df['start_timestamp'].dt.dayofweek  # Extract day of the week

    
    # Technical Indicators
    df['sma_10'] = df['close'].rolling(window=10).mean()  # Simple Moving Average (SMA)
    df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()  # Exponential Moving Average (EMA)
    
    # Momentum indicators
    df['rsi'] = compute_rsi(df['close'])  # Compute RSI

    # Volatility Indicator (Bollinger Bands)
    df['bollinger_high'] = df['sma_10'] + 2 * df['close'].rolling(window=10).std()
    df['bollinger_low'] = df['sma_10'] - 2 * df['close'].rolling(window=10).std()
    
    # Fill NaN values created by rolling calculations
    df.fillna(method='bfill', inplace=True)
    
    return df

def compute_rsi(series, period=14):
    """
    Compute the Relative Strength Index (RSI) for a given price series.

    Parameters:
    series (Series): The closing price series.
    period (int): The period for RSI calculation.

    Returns:
    Series: The RSI values.
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    return 100 - (100 / (1 + rs))

def train_or_update_linear_model(df, symbol, interval):
    """
    Train a new model if it doesn't exist or update an existing model with new data.

    Parameters:
    df (DataFrame): The DataFrame containing historical price data.
    symbol (str): The asset symbol.
    interval (str): The time interval for the model.

    Returns:
    None
    """
    MODEL_KEY_LINEAR = f'Mockba/technical_analysis_trained_model/{symbol}_{interval}_linear_model.joblib'
    local_model_path_linear = f'temp/{symbol}_{interval}_linear_model.joblib'

    # Ensure the directory exists
    os.makedirs(os.path.dirname(local_model_path_linear), exist_ok=True)

    # Add new features to the dataset
    df = add_features(df)
    
    # Select features for training
    features = ['sma_10', 'ema_10', 'rsi', 'bollinger_high', 'bollinger_low']
    X = df[features].values
    y = df['close'].values

    # Normalize feature values
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Check if the model already exists
    if download_model(BUCKET_NAME, MODEL_KEY_LINEAR, local_model_path_linear):
        # Load existing model
        model, existing_scaler = joblib.load(local_model_path_linear)
        print(f"Updating existing model for {symbol} on {interval} interval.")
        
        # Append new data to existing training data
        X_old = existing_scaler.transform(X)  # Use the previous scaler
        X_combined = np.vstack((X_old, X_scaled))
        y_combined = np.hstack((y, y))

        # Re-train the model
        model.fit(X_combined, y_combined)
    else:
        # Train new model
        print(f"Training new model for {symbol} on {interval} interval.")
        model = LinearRegression()
        model.fit(X_scaled, y)

    # Save the updated model and scaler
    joblib.dump((model, scaler), local_model_path_linear, compress=3)

    # Upload the model to the bucket
    upload_model(BUCKET_NAME, MODEL_KEY_LINEAR, local_model_path_linear)

    # Delete the local file after uploading
    if os.path.exists(local_model_path_linear):
        os.remove(local_model_path_linear)
    else:
        print(f"Local file {local_model_path_linear} does not exist.") 

############################################################################################################
# ARIMA Model
############################################################################################################

def test_stationarity(timeseries):
    """Perform the Augmented Dickey-Fuller test to check stationarity."""
    result = adfuller(timeseries)
    return result[1]  # p-value

def find_best_arima_params(timeseries, p_values, d_values, q_values):
    """Perform a grid search to find the best ARIMA parameters."""
    best_aic, best_cfg = float("inf"), None

    def evaluate_arima_order(order):
        p, d, q = order
        try:
            model = ARIMA(timeseries, order=(p, d, q))
            model_fit = model.fit()
            return model_fit.aic, (p, d, q)
        except:
            return float("inf"), None

    order_combinations = [(p, d, q) for p in p_values for d in d_values for q in q_values]

    with ThreadPoolExecutor(max_workers=cpu_count) as executor:
        results = executor.map(evaluate_arima_order, order_combinations)

    for aic, cfg in results:
        if cfg and aic < best_aic:
            best_aic, best_cfg = aic, cfg

    return best_cfg

def train_or_update_arima(df, symbol, interval):
    """
    Train a new ARIMA model if it doesn't exist or update an existing model with new data.

    Parameters:
    df (DataFrame): The DataFrame containing historical price data.
    symbol (str): The asset symbol.
    interval (str): The time interval for the model.

    Returns:
    None
    """
    MODEL_KEY_ARIMA = f'Mockba/technical_analysis_trained_model/{symbol}_{interval}_arima_model.joblib'
    model_filename = f'temp/{symbol}_{interval}_arima_model.joblib'
    
    # Extract prices and apply scaling (for better model stability)
    prices = df['close'].values.astype(float)  # Ensure data is float
    scaling_factor = 1e8 if prices.mean() < 1 else 1
    scaled_prices = prices * scaling_factor  # Scale up prices

    # Check if series is stationary
    d = 1 if test_stationarity(scaled_prices) > 0.05 else 0

    if download_model(BUCKET_NAME, MODEL_KEY_ARIMA, model_filename):
        # Load existing model
        model_fit = joblib.load(model_filename)
        print(f"Updating existing ARIMA model for {symbol} on {interval} interval.")

        # Retrieve the stored order correctly
        model_order = getattr(model_fit.model, 'order', None)
        if model_order is None:
            raise ValueError("Failed to retrieve ARIMA order from saved model.")

        # Extend the dataset and ensure it's a 1D NumPy array
        history = np.array(model_fit.model.endog, dtype=float)  # Get existing training data
        new_data = np.array(scaled_prices, dtype=float)  # Ensure new data is also float

        # Flatten history if it's 2D
        history = history.flatten()

        # Flatten new_data if it's 2D (just in case)
        new_data = new_data.flatten()

        # Append new prices properly
        history = np.concatenate((history, new_data))

        # Re-train the ARIMA model with updated dataset
        model = ARIMA(history, order=model_order)
        model_fit = model.fit()
    
    else:
        # Train a new model
        print(f"Training new ARIMA model for {symbol} on {interval} interval.")
        
        p_values = range(0, 6)
        d_values = [d]
        q_values = range(0, 6)

        best_cfg = find_best_arima_params(scaled_prices, p_values, d_values, q_values)
        if best_cfg is None:
            raise ValueError("No valid ARIMA configuration found.")

        model = ARIMA(scaled_prices, order=best_cfg)
        model_fit = model.fit()

    # Save the updated model
    joblib.dump(model_fit, model_filename, compress=3)

    # Upload the model to the bucket
    upload_model(BUCKET_NAME, MODEL_KEY_ARIMA, model_filename)

    # Delete the local file after uploading
    if os.path.exists(model_filename):
        os.remove(model_filename)
    else:
        print(f"Local file {model_filename} does not exist.") 

    print(f"✅ ARIMA model saved for {symbol} on {interval} interval.")

############################################################################################################
#XGBoost Model
############################################################################################################
def create_features(prices):
    """Generate features for XGBoost model."""
    df = pd.DataFrame({'close': prices})
    df['sma_10'] = df['close'].rolling(window=10).mean()
    df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['volatility'] = df['close'].rolling(window=10).std()
    df.fillna(method='bfill', inplace=True)  # Fill NaN values
    return df[['sma_10', 'ema_10', 'volatility']].values

def train_or_update_xgboost(df, symbol, interval):
    """
    Train a new XGBoost model if it doesn't exist or update an existing model with new data.

    Parameters:
    df (DataFrame): The DataFrame containing historical price data.
    symbol (str): The asset symbol.
    interval (str): The time interval for the model.

    Returns:
    None
    """
    MODEL_KEY_XGBOOST = f'Mockba/technical_analysis_trained_model/{symbol}_{interval}_xgboost_model.joblib'
    model_filename = f'temp/{symbol}_{interval}_xgboost_model.joblib'

    # Extract prices and normalize
    prices = df['close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_prices = scaler.fit_transform(prices).flatten()

    # Create features and training data
    features = create_features(scaled_prices)
    time_index = np.arange(len(scaled_prices)).reshape(-1, 1)
    X = np.hstack((time_index, features))
    y = scaled_prices

    if download_model(BUCKET_NAME, MODEL_KEY_XGBOOST, model_filename):
        # Load existing model
        best_model = joblib.load(model_filename)
        print(f"Updating existing XGBoost model for {symbol} on {interval} interval.")

        # Extend data for re-training
        latest_price = np.array([scaled_prices[-1]])  # Ensure it's a NumPy array
        new_features = create_features(np.append(scaled_prices, latest_price))[-1:]  # Take only last row

        new_time_index = np.array([[len(scaled_prices)]])  # Predict for the next time step

        # Ensure dimensions match before stacking
        if new_time_index.shape[0] != new_features.shape[0]:
            raise ValueError(f"Dimension mismatch: new_time_index {new_time_index.shape} vs new_features {new_features.shape}")

        new_X = np.hstack((new_time_index, new_features))  # Add new time step
        new_y = np.array([scaled_prices[-1]])  # Ensure it's a 1D array of length 1

        # Verify that the dimensions of X_combined and y_combined match
        X_combined = np.vstack((X, new_X))
        y_combined = np.hstack((y, new_y))

        if X_combined.shape[0] != y_combined.shape[0]:
            raise ValueError(f"Shape mismatch: X_combined {X_combined.shape} vs y_combined {y_combined.shape}")

        # Re-train the model
        best_model.fit(X_combined, y_combined)
    
    else:
        # Train new model
        print(f"Training new XGBoost model for {symbol} on {interval} interval.")

        param_dist = {
            'n_estimators': randint(100, 500),
            'learning_rate': [0.01, 0.05, 0.1], 
            'max_depth': [3, 5, 7, 10],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.3, 0.5, 0.7, 1.0],
            'gamma': [0, 0.1, 0.2, 0.3],
            'min_child_weight': [1, 3, 5, 7],
            'reg_alpha': [0, 0.01, 0.1, 1],
            'reg_lambda': [1, 1.5, 2, 3]
        }

        model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
        random_search = RandomizedSearchCV(
            estimator=model, 
            param_distributions=param_dist, 
            n_iter=50,  # Faster training
            cv=3,  
            scoring='neg_mean_squared_error', 
            random_state=42,
            n_jobs=cpu_count
        )
        random_search.fit(X, y)

        # Best model
        best_model = xgb.XGBRegressor(**random_search.best_params_, objective='reg:squarederror', random_state=42)
        best_model.fit(X, y)

    # Save the updated model
    joblib.dump(best_model, model_filename, compress=3)

    # Upload the model to the bucket
    upload_model(BUCKET_NAME, MODEL_KEY_XGBOOST, model_filename)

    # Delete the local file after uploading
    if os.path.exists(model_filename):
        os.remove(model_filename)
    else:
        print(f"Local file {model_filename} does not exist.") 

    print(f"✅ XGBoost model saved for {symbol} on {interval} interval.")

def train_technical_models(symbol, interval):
    """
    Train all technical analysis models for a given symbol and interval.

    Parameters:
    symbol (str): The asset symbol.
    interval (str): The time interval for the model.

    Returns:
    None
    """
    df = get_historical_data_limit(symbol, interval, limit=500)
    train_or_update_linear_model(df, symbol, interval)
    train_or_update_arima(df, symbol, interval)
    train_or_update_xgboost(df, symbol, interval)

# Main function to train or update models for multiple intervals
def train_models(symbol, intervals):
    for interval in intervals:
        train_technical_models(symbol, interval)      

# if __name__ == "__main__":
#     symbol = 'APTUSDT'
#     intervals = ["1h", "4h", "1d"]
#     train_technical_models(symbol, interval)
#     print("Training completed.")
