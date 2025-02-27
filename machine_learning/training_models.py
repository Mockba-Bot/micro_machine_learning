import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint
import psycopg2  # Library for PostgreSQL connection
from dotenv import load_dotenv
import joblib  # Library for model serialization
from datetime import datetime, timedelta  # Import timedelta from datetime
from sqlalchemy import text
import requests
from datetime import timedelta
# Add the directory containing your modules to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from database import getHistorical
from database import operations
from historical_data import get_historical_data
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


# Add technical indicators to the data
def add_indicators(data):
    """Adds advanced technical indicators to a price dataset."""

    # Ensure the columns are of numeric type
    data[['close', 'high', 'low', 'volume']] = data[['close', 'high', 'low', 'volume']].apply(pd.to_numeric)

    # --- Relative Strength Index (RSI) ---
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    data['rsi'] = 100 - (100 / (1 + rs))

    # --- Moving Average Convergence Divergence (MACD) ---
    data['ema_12'] = data['close'].ewm(span=12, adjust=False).mean()
    data['ema_26'] = data['close'].ewm(span=26, adjust=False).mean()
    data['macd'] = data['ema_12'] - data['ema_26']
    data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
    data['macd_diff'] = data['macd'] - data['macd_signal']

    # --- Bollinger Bands ---
    data['bollinger_mavg'] = data['close'].rolling(window=20).mean()
    data['bollinger_std'] = data['close'].rolling(window=20).std()
    data['bollinger_hband'] = data['bollinger_mavg'] + (data['bollinger_std'] * 2)
    data['bollinger_lband'] = data['bollinger_mavg'] - (data['bollinger_std'] * 2)

    # --- Exponential Moving Averages (EMA) ---
    data['ema_21'] = data['close'].ewm(span=21, adjust=False).mean()
    data['ema_50'] = data['close'].ewm(span=50, adjust=False).mean()
    data['ema_200'] = data['close'].ewm(span=200, adjust=False).mean()

    # --- Simple Moving Averages (SMA) ---
    data['sma_50'] = data['close'].rolling(window=50).mean()
    data['sma_200'] = data['close'].rolling(window=200).mean()

    # --- Stochastic Oscillator ---
    data['stoch_k'] = ((data['close'] - data['low'].rolling(14).min()) /
                       (data['high'].rolling(14).max() - data['low'].rolling(14).min())) * 100
    data['stoch_d'] = data['stoch_k'].rolling(3).mean()

    # --- Average True Range (ATR) ---
    data['tr'] = pd.concat([
        data['high'] - data['low'],
        (data['high'] - data['close'].shift()).abs(),
        (data['low'] - data['close'].shift()).abs()
    ], axis=1).max(axis=1)
    data['atr'] = data['tr'].rolling(window=14).mean()

    # --- VWAP (Volume Weighted Average Price) ---
    data['vwap'] = (data['volume'] * (data['high'] + data['low'] + data['close']) / 3).cumsum() / data['volume'].cumsum()

    # --- ADX (Average Directional Index) ---
    plus_dm = data['high'].diff()
    minus_dm = data['low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    atr = data['atr']
    plus_di = 100 * (plus_dm.ewm(span=14, adjust=False).mean() / atr)
    minus_di = abs(100 * (minus_dm.ewm(span=14, adjust=False).mean() / atr))
    adx = 100 * abs((plus_di - minus_di) / (plus_di + minus_di)).rolling(14).mean()
    data['adx'] = adx

    # --- Commodity Channel Index (CCI) ---
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    mean_deviation = typical_price.rolling(window=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
    data['cci'] = (typical_price - typical_price.rolling(window=20).mean()) / (0.015 * mean_deviation)

    # --- Williams %R (Momentum Indicator) ---
    data['williams_r'] = (data['high'].rolling(14).max() - data['close']) / \
                          (data['high'].rolling(14).max() - data['low'].rolling(14).min()) * -100

    # --- Momentum Indicator ---
    data['momentum'] = data['close'].diff(periods=10)

    # --- Rate of Change (ROC) ---
    data['roc'] = data['close'].pct_change(periods=10) * 100

    # --- Parabolic SAR (Stop and Reverse) ---
    data['sar'] = np.nan
    af = 0.02  # Acceleration Factor
    max_af = 0.2
    ep = data['high'][0]  # Extreme point
    sar = data['low'][0]  # Start SAR with first low
    trend = 1  # 1 = uptrend, -1 = downtrend
    for i in range(1, len(data)):
        prev_sar = sar
        sar = prev_sar + af * (ep - prev_sar)
        if trend == 1:
            if data['low'][i] < sar:
                trend = -1
                sar = ep
                ep = data['low'][i]
                af = 0.02
        else:
            if data['high'][i] > sar:
                trend = 1
                sar = ep
                ep = data['high'][i]
                af = 0.02
        if af < max_af:
            af += 0.02
        data.loc[data.index[i], 'sar'] = sar

    # --- Ichimoku Cloud Components ---
    data['tenkan_sen'] = (data['high'].rolling(window=9).max() + data['low'].rolling(window=9).min()) / 2
    data['kijun_sen'] = (data['high'].rolling(window=26).max() + data['low'].rolling(window=26).min()) / 2
    data['senkou_span_a'] = ((data['tenkan_sen'] + data['kijun_sen']) / 2).shift(26)
    data['senkou_span_b'] = ((data['high'].rolling(window=52).max() + data['low'].rolling(window=52).min()) / 2).shift(26)
    data['chikou_span'] = data['close'].shift(-26)

    # Fill NaN values after calculations
    data.fillna(method='bfill', inplace=True)

    return data

# Train the machine learning model with advanced hyperparameter tuning
def train_models_swing(token, model_path):
    MODEL_KEY = f'Mockba/trained_models/trained_model_{pair}_{timeframe}.pkl'
    local_model_path = f'temp/trained_model_{pair}_{timeframe}.pkl'

    # Get the current date
    now = datetime.now()
    current_date = now.strftime('%Y-%m-%d')
    values = f'2024-01-01|{current_date}'

    # Fetch historical data
    getHistorical.get_all_binance(pair, timeframe, token, save=True)

    # Get historical data
    data = get_historical_data(pair, timeframe, values)

    # Add technical indicators
    data = add_indicators(data)

    # Automatically determine the feature columns (Exclude non-numeric ones)
    exclude_columns = ['timestamp', 'return', 'target', 'tr']
    features = [col for col in data.columns if col not in exclude_columns]

    print(f"Using Features: {features}")

    # Define target variable
    data['target'] = (data['close'].shift(-1) > data['close']).astype(int)

    # Drop NaN values
    data = data.dropna(subset=features + ['target'])

    print("Train Model")

    # Check if the model exists in storage
    if download_model(BUCKET_NAME, MODEL_KEY, local_model_path):
        # Load existing model
        print("Loaded existing model.")
        model = joblib.load(local_model_path)
        update_model(model, data, features)
        upload_model(BUCKET_NAME, MODEL_KEY, local_model_path)
    else:
        # Train a new model if none exists
        print("No existing model found. Training a new model.")
        train_model(data, local_model_path, features)
        upload_model(BUCKET_NAME, MODEL_KEY, local_model_path)

    # Delete local model file after upload
    if os.path.exists(local_model_path):
        os.remove(local_model_path)
    else:
        print(f"Local file {local_model_path} does not exist.")

    print("✅ Model training complete.")    

# Update the existing model with new data
def update_model(existing_model, new_data, features):
    # Calculate return and target columns for the new data
    new_data['return'] = new_data['close'].pct_change().shift(-1)
    new_data['target'] = (new_data['return'] > 0).astype(int)
    
    # Prepare the new data for training
    X_new = new_data[features].dropna()
    y_new = new_data['target'].dropna().loc[X_new.index]
    
    # Update the existing model with the new data
    existing_model.fit(X_new, y_new)

# Train the machine learning models for all currency pairs and timeframes
def train_models_swing(token, pair, timeframe):
    MODEL_KEY = f'Mockba/trained_models/trained_model_{pair}_{timeframe}.pkl'
    local_model_path = f'temp/trained_model_{pair}_{timeframe}.pkl'
    # Get the current date
    now = datetime.now()
    current_date = now.strftime('%Y-%m-%d')
    values = f'2024-01-01|{current_date}'
    # Get the model path to verify if the model already exists
    # Fetch historical data and add technical indicators
    getHistorical.get_all_binance(pair, timeframe, token, save=True)
    # Get the historical data for the pair and timeframe
    data = get_historical_data(pair, timeframe, values)
    data = add_indicators(data)
    # Check if the model exists in DigitalOcean Spaces
    if download_model(BUCKET_NAME, MODEL_KEY, local_model_path):
        # Load the existing model
        print("Loaded existing model.")
        model = joblib.load(local_model_path)
        features = ['rsi', 'macd', 'macd_signal', 'macd_diff', 'bollinger_hband', 'bollinger_mavg', 'bollinger_lband', 'ema', 'ATR']
        update_model(model, data, features)
        upload_model(BUCKET_NAME, MODEL_KEY, local_model_path)
    else:
        # Train a new model if it doesn't exist
        print("No existing model found. Training a new model.")
        train_model(data, local_model_path)
        upload_model(BUCKET_NAME, MODEL_KEY, local_model_path)

    # Delete the local file after uploading
    if os.path.exists(local_model_path):
        os.remove(local_model_path)
    else:
        print(f"Local file {local_model_path} does not exist.")
    print("Model training complete.")    

# Main function to train or update models for multiple intervals
def train_models(symbol, intervals):
    for interval in intervals:
        train_models_swing('000000', symbol, interval)

# if __name__ == "__main__":
      # intervals = ["1h", "4h", "1d"]
#     train_models_swing('000000', 'APTUSDT', intervals)