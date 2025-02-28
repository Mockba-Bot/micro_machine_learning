import os
import sys
import joblib
import numpy as np
import pandas as pd
import requests
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
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
cpu_count = os.cpu_count()-int(CPU_COUNT)
BUCKET_NAME = os.getenv("BUCKET_NAME")  # Your bucket name


# Adjust limit based on interval
def get_limit_for_interval(interval):
    return {
        '1h': 500,
        '4h': 200,
        '1d': 100
    }.get(interval, 500)  # Default to 500 if interval is unknown

# Function to calculate RSI manually
def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = pd.Series(gain).rolling(window=window).mean()
    avg_loss = pd.Series(loss).rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to calculate MACD manually
def calculate_macd(series, slow=26, fast=12, signal=9):
    fast_ema = series.ewm(span=fast, min_periods=fast).mean()
    slow_ema = series.ewm(span=slow, min_periods=slow).mean()
    macd = fast_ema - slow_ema
    macd_signal = macd.ewm(span=signal, min_periods=signal).mean()
    return macd, macd_signal

# Function to calculate Bollinger Bands manually
def calculate_bollinger_bands(series, window=20, num_std=2):
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    
    return upper_band, rolling_mean, lower_band

# Function to add technical indicators
def add_technical_indicators(df):
    print(f"Adding technical indicators for {df['start_timestamp'].iloc[0].date()} - {df['start_timestamp'].iloc[-1].date()}")
    
    try:
        df['rsi'] = calculate_rsi(df['close'])
        df['macd'], df['macd_signal'] = calculate_macd(df['close'])
        df['bollinger_hband'], df['bollinger_mavg'], df['bollinger_lband'] = calculate_bollinger_bands(df['close'])
        df['ema'] = df['close'].ewm(span=14, adjust=False).mean()  # Exponential Moving Average
        df['volatility'] = df['close'].rolling(window=10).std()  # Rolling Standard Deviation
        df['price_rate_of_change'] = df['close'].pct_change()  # Percentage Change
        
        df.fillna(method='bfill', inplace=True)  # Fill NaN values
        
    except Exception as e:
        print(f"Error while adding technical indicators: {e}")
        raise

    return df

# Train or update the ensemble model
def train_or_update_signal_model(symbol, interval):
    MODEL_KEY_SIGNAL = f'Mockba/signal_models/{symbol}_{interval}_signal_model.pkl'
    model_path = f'temp/{symbol}_{interval}_signal_model.joblib'

    # Get historical data
    limit = get_limit_for_interval(interval)
    data = get_historical_data_limit(symbol, interval, limit)

    # Add technical indicators
    data = add_technical_indicators(data)

    # Define target variable (binary classification)
    data['target'] = (data['close'].shift(-1) > data['close']).astype(int)

    # Define features (removed 'doji', 'hammer', 'engulfing')
    features = ['rsi', 'macd', 'macd_signal', 'bollinger_hband', 'bollinger_mavg', 'bollinger_lband', 
                'ema', 'volatility', 'price_rate_of_change']

    # Drop rows where any feature column or target column has NaN values
    valid_features = [col for col in features if col in data.columns]  # Ensure only existing features are used

    data = data.dropna(subset=valid_features + ['target'])
    
    X = data[features]
    y = data['target']

    # Define parameter distributions for RandomizedSearchCV
    param_dist_rf = {
        'n_estimators': randint(50, 300),
        'max_depth': randint(10, 60),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 20),
        'max_features': ['sqrt', 'log2']
    }
    
    param_dist_xgb = {
        'n_estimators': randint(50, 300),
        'max_depth': randint(3, 8),
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.5, 0.7, 1.0],
    }

    cpu_count = os.cpu_count() - 1

    if download_model(BUCKET_NAME, MODEL_KEY_SIGNAL, model_path):
        # Load existing model and update it
        ensemble_model = joblib.load(model_path)
        print(f"Updating existing signal model for {symbol} on {interval} interval.")
        ensemble_model.fit(X, y)
    else:
        # Train new model
        print(f"Training new signal model for {symbol} on {interval} interval.")

        rf = RandomForestClassifier(random_state=42)
        rf_random = RandomizedSearchCV(rf, param_distributions=param_dist_rf, n_iter=200, cv=5, n_jobs=cpu_count, random_state=42)
        rf_random.fit(X, y)

        xgb_clf = xgb.XGBClassifier(random_state=42)
        xgb_random = RandomizedSearchCV(xgb_clf, param_distributions=param_dist_xgb, n_iter=200, cv=5, n_jobs=cpu_count, random_state=42)
        xgb_random.fit(X, y)

        ensemble_model = VotingClassifier(estimators=[
            ('rf', rf_random.best_estimator_), ('xgb', xgb_random.best_estimator_)
        ], voting='soft')
        ensemble_model.fit(X, y)

    # Save the updated model
    joblib.dump(ensemble_model, model_path)

    upload_model(BUCKET_NAME, MODEL_KEY_SIGNAL, model_path)

    # Delete the local file after uploading
    if os.path.exists(model_path):
        os.remove(model_path)
    else:
        print(f"Local file {model_path} does not exist.")

# Main function to train or update models for multiple intervals
def train_models(symbol, intervals):
    for interval in intervals:
        train_or_update_signal_model(symbol, interval)

# Run training for selected intervals
# if __name__ == "__main__":
#     symbol = "APTUSDT"
#     intervals = ["1h", "4h", "1d"]
#     train_signal_models(symbol, intervals)
