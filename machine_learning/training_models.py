import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint
import psycopg2  # Library for PostgreSQL connection
from dotenv import load_dotenv
import numpy as np
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
def add_indicators(data, required_features):
    """
    Adds only the technical indicators required by the trained model.

    Parameters:
        data (pd.DataFrame): The input DataFrame containing price data.
        required_features (list): List of feature names that the model requires.

    Returns:
        pd.DataFrame: The DataFrame with the required indicators added.
    """

    # Ensure the columns are of numeric type
    data[['close', 'high', 'low', 'volume']] = data[['close', 'high', 'low', 'volume']].apply(pd.to_numeric)

    # --- Relative Strength Index (RSI) ---
    if 'rsi' in required_features:
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))

    # --- Moving Average Convergence Divergence (MACD) ---
    if any(x in required_features for x in ['macd', 'macd_signal', 'macd_diff']):
        data['ema_12'] = data['close'].ewm(span=12, adjust=False).mean()
        data['ema_26'] = data['close'].ewm(span=26, adjust=False).mean()
        data['macd'] = data['ema_12'] - data['ema_26']
        data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
        data['macd_diff'] = data['macd'] - data['macd_signal']

    # --- Bollinger Bands ---
    if any(x in required_features for x in ['bollinger_mavg', 'bollinger_std', 'bollinger_hband', 'bollinger_lband']):
        data['bollinger_mavg'] = data['close'].rolling(window=20).mean()
        data['bollinger_std'] = data['close'].rolling(window=20).std()
        data['bollinger_hband'] = data['bollinger_mavg'] + (data['bollinger_std'] * 2)
        data['bollinger_lband'] = data['bollinger_mavg'] - (data['bollinger_std'] * 2)

    # --- Exponential Moving Averages (EMA) ---
    for ema in ['ema_20', 'ema_50', 'ema_200']:
        if ema in required_features:
            span = int(ema.split('_')[1])
            data[ema] = data['close'].ewm(span=span, adjust=False).mean()

    # --- Simple Moving Averages (SMA) ---
    for sma in ['sma_5', 'sma_20', 'sma_50', 'sma_200']:
        if sma in required_features:
            window = int(sma.split('_')[1])
            data[sma] = data['close'].rolling(window=window).mean()

    # --- Stochastic Oscillator ---
    if any(x in required_features for x in ['stoch_k', 'stoch_d']):
        data['stoch_k'] = ((data['close'] - data['low'].rolling(14).min()) /
                           (data['high'].rolling(14).max() - data['low'].rolling(14).min())) * 100
        data['stoch_d'] = data['stoch_k'].rolling(3).mean()

    # --- Average True Range (ATR) ---
    if 'atr' in required_features:
        data['tr'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - data['close'].shift()).abs(),
            (data['low'] - data['close'].shift()).abs()
        ], axis=1).max(axis=1)
        data['atr'] = data['tr'].rolling(window=14).mean()

    # --- VWAP (Volume Weighted Average Price) ---
    if 'vwap' in required_features:
        data['vwap'] = (data['volume'] * (data['high'] + data['low'] + data['close']) / 3).cumsum() / data['volume'].cumsum()

    # --- ADX (Average Directional Index) ---
    if 'adx' in required_features:
        plus_dm = data['high'].diff()
        minus_dm = data['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        atr = data['atr'] if 'atr' in data.columns else data['close'].rolling(window=14).mean()
        plus_di = 100 * (plus_dm.ewm(span=14, adjust=False).mean() / atr)
        minus_di = abs(100 * (minus_dm.ewm(span=14, adjust=False).mean() / atr))
        data['adx'] = 100 * abs((plus_di - minus_di) / (plus_di + minus_di)).rolling(14).mean()

    # --- Commodity Channel Index (CCI) ---
    if 'cci' in required_features:
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        mean_deviation = typical_price.rolling(window=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
        data['cci'] = (typical_price - typical_price.rolling(window=20).mean()) / (0.015 * mean_deviation)

    # --- Williams %R (Momentum Indicator) ---
    if 'williams_r' in required_features:
        data['williams_r'] = (data['high'].rolling(14).max() - data['close']) / \
                              (data['high'].rolling(14).max() - data['low'].rolling(14).min()) * -100

    # --- Momentum Indicator ---
    if 'momentum' in required_features:
        data['momentum'] = data['close'].diff(periods=10)

    # --- Rate of Change (ROC) ---
    if 'roc' in required_features:
        data['roc'] = data['close'].pct_change(periods=10) * 100

    # --- Parabolic SAR (Stop and Reverse) ---
    if 'sar' in required_features:
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
    if any(x in required_features for x in ['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span']):
        data['tenkan_sen'] = (data['high'].rolling(window=9).max() + data['low'].rolling(window=9).min()) / 2
        data['kijun_sen'] = (data['high'].rolling(window=26).max() + data['low'].rolling(window=26).min()) / 2
        data['senkou_span_a'] = ((data['tenkan_sen'] + data['kijun_sen']) / 2).shift(26)
        data['senkou_span_b'] = ((data['high'].rolling(window=52).max() + data['low'].rolling(window=52).min()) / 2).shift(26)
        data['chikou_span'] = data['close'].shift(-26)

    # Fill NaN values after calculations
    data.fillna(method='bfill', inplace=True)

    return data

# Train the machine learning model with advanced hyperparameter tuning
def train_model(data, model_path, features):
    # Calculate return and target columns
    data['return'] = data['close'].pct_change().shift(-1)
    data['target'] = (data['return'] > 0).astype(int)
    
    # Prepare training and testing datasets
    X = data[features].dropna()
    y = data['target'].dropna().loc[X.index]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define hyperparameter search space
    param_distributions = {
        'n_estimators': randint(100, 500),
        'max_depth': randint(10, 80),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 20),
        'max_features': ['sqrt', 'log2']
    }
    
    # Perform randomized hyperparameter search
    randomized_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42), 
        param_distributions, 
        n_iter=300,  # Increased number of iterations
        cv=3, 
        scoring='accuracy', 
        random_state=42,
        n_jobs=cpu_count
    )
    randomized_search.fit(X_train, y_train)
    
    # Get the best model from the search
    best_model = randomized_search.best_estimator_

    # Save the trained model to disk
    joblib.dump(best_model, model_path)


# Update the existing model with new data
def update_model(existing_model, new_data, features):
    """
    Update the existing model with new data using the `warm_start` approach.
    """
    # Calculate return and target columns
    new_data['return'] = new_data['close'].pct_change().shift(-1)
    new_data['target'] = (new_data['return'] > 0).astype(int)

    # Prepare the dataset
    X_new = new_data[features].dropna()
    y_new = new_data['target'].dropna().loc[X_new.index]

    # Ensure `existing_model` is a RandomForestClassifier
    if isinstance(existing_model, RandomForestClassifier):
        existing_model.n_estimators += 50  # Add 50 new trees instead of retraining
        existing_model.set_params(warm_start=True)
        existing_model.fit(X_new, y_new)
    else:
        raise ValueError("Expected a RandomForestClassifier model for incremental training")

    return existing_model


# Train the machine learning model with advanced hyperparameter tuning
def train_machine_learning(pair, timeframe, features=None):
    model = "_".join(features).replace("[", "").replace("]", "").replace("'", "_").replace(" ", "")
    MODEL_KEY = f'Mockba/trained_models/trained_model_{pair}_{timeframe}_{model}.pkl'
    local_model_path = f'temp/trained_model_{pair}_{timeframe}_{model}.pkl'

    # Get the current date
    now = datetime.now()
    current_date = now.strftime('%Y-%m-%d')
    values = f'2024-01-01|{current_date}'

    # Get historical data
    data = get_historical_data(pair, timeframe, values)

    # Add technical indicators
    data = add_indicators(data, features)

    # Automatically determine the feature columns (Exclude non-numeric ones)
    exclude_columns = ['start_timestamp']
    features = [col for col in data.columns if col not in exclude_columns]

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


# Main function to train or update models for multiple intervals
def train_models(symbol, intervals, features):
    for interval in intervals:
        train_machine_learning(symbol, interval, features)

# Trend-following + volatility breakout strategy, Best for swing trading or momentum-based scalping
features =  ["rsi", "macd", "macd_signal", "macd_diff", "bollinger_hband", "bollinger_mavg", "bollinger_lband", "ema_20", "atr"]

# Momentum & Trend Following Strategy, Best for: Trend-following strategies, swing trading.
# "rsi" → Detects overbought/oversold conditions
# "macd" → Measures trend strength
# "macd_signal" → Helps identify buy/sell signals
# "ema_20" → Short-term trend filter
# "ema_50" → Medium-term trend confirmation
# "atr" → Measures volatility for stop-loss placement
features = ["rsi", "macd", "macd_signal", "ema_20", "ema_50", "atr"]  

# Mean Reversion Strategy, Best for: Buying dips, selling peaks.    
# "bollinger_mavg" → Identifies mean price
# "bollinger_hband" → Upper resistance level
# "bollinger_lband" → Lower support level
# "rsi" → Confirms oversold/overbought conditions
# "cci" → Measures price deviation from mean
# "atr" → Identifies breakout volatility 
features = ["bollinger_mavg", "bollinger_hband", "bollinger_lband", "rsi", "cci", "atr"]

# Volatility Breakout Strategy, Best for: Trading strong moves in volatile markets.
# "atr" → Measures breakout strength
# "bollinger_hband" → Breakout confirmation
# "bollinger_lband" → Breakdown confirmation
# "adx" → Measures trend strength
# "momentum" → Confirms breakout momentum
# "ema_50" → Medium-term trend confirmation
features = ["atr", "bollinger_hband", "bollinger_lband", "adx", "momentum", "ema_50"]

# Reversal Detection Strategy, Best for: Catching market tops/bottoms.
# "williams_r" → Identifies reversal points
# "rsi" → Confirms overbought/oversold conditions
# "macd" → Checks trend direction
# "macd_signal" → Identifies crossovers
# "bollinger_mavg" → Defines fair price range
# "roc" → Measures price acceleration
features = ["williams", "rsi", "macd", "macd_signal", "bollinger_mavg", "roc"]

#  Institutional Strategy (Advanced), Best for: Large volume traders, institutions.
# "vwap" → Confirms institutional price levels
# "ema_50" → Identifies trend direction
# "ema_200" → Long-term trend filter
# "adx" → Confirms trend strength
# "sar" → Provides stop/reversal points
# "senkou_span_a" → Ichimoku support/resistance
features = ["vwap", "ema_50", "ema_200", "adx", "sar", "senkou_span_a"]

# if __name__ == "__main__":
#     intervals = ["1h", "4h", "1d"]
#     train_models('PERP_APT_USDC', intervals, features)