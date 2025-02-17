import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, RandomForestRegressor
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
from bucket import download_model, upload_model
import warnings
warnings.filterwarnings("ignore")
# Load environment variables from the .env file
dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env.micro.machine.learning'))
load_dotenv(dotenv_path=dotenv_path)

# Access the environment variables
CPU_COUNT = os.getenv("CPU_COUNT")
cpu_count = os.cpu_count()-int(CPU_COUNT)
BUCKET_NAME = os.getenv("BUCKET_NAME")  # Your bucket name

# Fetch historical data from the database
def get_historical_data(pair, timeframe, values):
    field = '"timestamp"'
    table = f'"{pair}_{timeframe}"'
    f, t = values.split('|')
    
    query = text(f"""
        SELECT DISTINCT {field}, low, high, volume, close 
        FROM public.{table} 
        WHERE timestamp >= :start_time AND timestamp <= :end_time 
        ORDER BY 1
    """)
    
    df = pd.read_sql(query, con=operations.db_con, params={"start_time": f, "end_time": t})
    
    # Convert columns to numeric types
    df['close'] = pd.to_numeric(df['close'])
    df['high'] = pd.to_numeric(df['high'])
    df['low'] = pd.to_numeric(df['low'])
    df['volume'] = pd.to_numeric(df['volume'])
    
    return df

# Function to calculate technical indicators
def calculate_indicators(df):
    """Calculate technical indicators for the given DataFrame without using TA-Lib."""
    
    # Create a copy of the DataFrame to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Fixing the conversion of 'high', 'low', and 'close' columns to numeric
    df['high'] = pd.to_numeric(df['high'], errors='coerce')
    df['low'] = pd.to_numeric(df['low'], errors='coerce')
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    
    # RSI (Relative Strength Index) - 5-period
    delta = df['close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=5, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(window=5, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-10)  # Adding small value to avoid division by zero
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD (Moving Average Convergence Divergence)
    short_ema = df['close'].ewm(span=12, adjust=False).mean()
    long_ema = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = short_ema - long_ema
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    
    # Exponential Moving Average (EMA)
    df['EMA5'] = df['close'].ewm(span=5, adjust=False).mean()
    
    # Average True Range (ATR) - Volatility indicator
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=10, min_periods=1).mean()
    
    # Momentum indicator (14-period)
    df['Momentum'] = df['close'].diff(periods=14)
    
    # Volume (if available)
    if 'volume' in df.columns:
        df['Volume'] = df['volume']  # Ensure the volume column is there
    
    # Fixing the removal of rows with NaN values
    df.dropna(inplace=True)
    
    return df


# Train a model for dynamic profit target prediction based on ATR and other features
def train_profit_target_model(df, model_path, profit_target_from, profit_target_to):

    df = calculate_indicators(df)  # Ensure the technical indicators are calculated

    # Feature selection for the profit target model
    X = df[['RSI', 'MACD', 'MACD_signal', 'Momentum', 'ATR', 'EMA5']].values

    # Create target (profit target) based on past gains (simulated for this purpose)
    y = np.clip(np.abs(df['close'].pct_change().shift(-1)) * 100, profit_target_from, profit_target_to)

    # Drop the last row from X to match the length of y
    X = X[:-1]
    y = y[:-1]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Optimized RandomForestRegressor setup
    model = RandomForestRegressor(
        n_estimators=100,  # Increased for better accuracy within available resources
        max_depth=cpu_count,       # Balanced depth for efficiency
        random_state=42,
        n_jobs=6           # Use 6 cores for parallel processing
    )

    model.fit(X_train, y_train)

    # Save the trained model to disk
    joblib.dump(best_model, model_path) 

# Update the existing model with new data if exists
def update_profit_target_model(existing_model, df, profit_target_from, profit_target_to):
    # Ensure the technical indicators are calculated
    df = calculate_indicators(df)

    # Feature selection for the profit target model
    X = df[['RSI', 'MACD', 'MACD_signal', 'Momentum', 'ATR', 'EMA5']].values

    # Create target (profit target) based on past gains (simulated for this purpose)
    y = np.clip(np.abs(df['close'].pct_change().shift(-1)) * 100, profit_target_from, profit_target_to)

    # Drop the last row from X to match the length of y
    X = X[:-1]
    y = y[:-1]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Get the number of CPU cores
    num_cores = cpu_count

    # Update the existing model with new data
    existing_model.set_params(
        warm_start=True,  # Enable warm_start to add more trees
        n_estimators=existing_model.n_estimators + 10,  # Add 10 more trees (adjust as needed)
        max_depth=num_cores,  # Set max_depth based on CPU cores (optional, but not recommended)
        n_jobs=num_cores - 1  # Use all but one core for parallel processing
    )

    # Retrain the model on the new data
    existing_model.fit(X_train, y_train)

    return existing_model

# Function to train a machine learning model
def train_model(df, model_path):
    df = df.dropna()  # Drop rows with NaN values from indicators
    # Define features and target
    X = df[['RSI', 'MACD', 'MACD_signal', 'Volume', 'Momentum']]
    y = np.where((df['close'].shift(-1) - df['close']) > 0, 1, 0)  # Binary target column

    # Align X and y lengths (both must have the same number of samples)
    X = X[:-1]
    y = y[:-1]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Hyperparameter tuning grid
    param_grid = {
        'n_estimators': [50, 100, 200],       # Adjusted for accuracy and resources
        'max_depth': [5, 10, 15],            # Increased range for depth
        'min_samples_split': [2, 5, 10],     # Ensure flexibility for splits
        'min_samples_leaf': [1, 2, 4],       # Regularization through leaf size
    }

    # GridSearch with RandomForestClassifier
    model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=cpu_count, verbose=1)
    grid_search.fit(X_train, y_train)

    # Retrieve the best model
    best_model = grid_search.best_estimator_

    # Save the trained model to disk
    joblib.dump(best_model, model_path)

# Function to update an existing model with new data
def update_model(existing_model, df):
    """
    Update an existing RandomForestClassifier model with new data.
    
    Parameters:
        existing_model (RandomForestClassifier): The existing model to update.
        df (pd.DataFrame): The DataFrame containing the new data.
    
    Returns:
        RandomForestClassifier: The updated model.
    """
    df = df.dropna()  # Drop rows with NaN values from indicators

    # Define features and target
    X = df[['RSI', 'MACD', 'MACD_signal', 'Volume', 'Momentum']]
    y = np.where((df['close'].shift(-1) - df['close']) > 0, 1, 0)  # Binary target column

    # Align X and y lengths (both must have the same number of samples)
    X = X[:-1]
    y = y[:-1]

    # Train-test split (optional, if you want to evaluate the updated model)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Get the number of CPU cores
    num_cores = cpu_count

    # Enable warm_start to add more trees
    existing_model.set_params(
        warm_start=True,  # Enable incremental training
        n_estimators=existing_model.n_estimators + 50,  # Add 50 more trees (adjust as needed)
        n_jobs=cpu_count  # Use all but one CPU core for parallel processing
    )

    # Retrain the model on the new data
    existing_model.fit(X_train, y_train)


# Train partial exit threshold model
def train_partial_exit_threshold_model(df, model_path, partial_exit_threshold_from, partial_exit_threshold_to):
    df = calculate_indicators(df)  # Ensure technical indicators are calculated

    # Feature selection for the partial exit threshold model
    X = df[['ATR', 'RSI']].values

    # Target for partial exit threshold
    y = np.clip(np.abs(df['close'].pct_change().shift(-1)) * 100, partial_exit_threshold_from, partial_exit_threshold_to)

    # Drop the last row from X and y to align their lengths
    X = X[:-1]
    y = y[:-1]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Optimized RandomForestRegressor setup
    model = RandomForestRegressor(
        n_estimators=100,  # Increased estimators for better accuracy
        max_depth=6,       # Depth optimized for available CPUs
        random_state=42,
        n_jobs=cpu_count           # Use 6 cores for parallel processing
    )

    model.fit(X_train, y_train)

    # Save the trained model to disk
    joblib.dump(model, model_path)

# Update partial exit threshold model
def update_partial_exit_threshold_model(existing_model, df, partial_exit_threshold_from, partial_exit_threshold_to):
    """
    Update an existing RandomForestRegressor model with new data for partial exit threshold prediction.
    
    Parameters:
        existing_model (RandomForestRegressor): The existing model to update.
        df (pd.DataFrame): The DataFrame containing the new data.
        partial_exit_threshold_from (float): The lower bound for the partial exit threshold.
        partial_exit_threshold_to (float): The upper bound for the partial exit threshold.
    
    Returns:
        RandomForestRegressor: The updated model.
    """
    df = calculate_indicators(df)  # Ensure technical indicators are calculated

    # Feature selection for the partial exit threshold model
    X = df[['ATR', 'RSI']].values

    # Target for partial exit threshold
    y = np.clip(np.abs(df['close'].pct_change().shift(-1)) * 100, partial_exit_threshold_from, partial_exit_threshold_to)

    # Drop the last row from X and y to align their lengths
    X = X[:-1]
    y = y[:-1]

    # Train-test split (optional, if you want to evaluate the updated model)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Get the number of CPU cores
    num_cores = cpu_count

    # Enable warm_start to add more trees
    existing_model.set_params(
        warm_start=True,  # Enable incremental training
        n_estimators=existing_model.n_estimators + 50,  # Add 50 more trees (adjust as needed)
        n_jobs=num_cores - 1  # Use all but one CPU core for parallel processing
    )

    # Retrain the model on the new data
    existing_model.fit(X_train, y_train)


# Train exit remaining percentage model
def train_exit_remaining_percentage_model(df, model_path, exit_remaining_percentage_from, exit_remaining_percentage_to):
    df = calculate_indicators(df)  # Ensure technical indicators are calculated

    # Feature selection for the exit remaining percentage model
    X = df[['MACD', 'MACD_signal', 'EMA5', 'RSI', 'Momentum', 'ATR', 'Volume']].values

    # Adjusted target calculation with a look-ahead window for multi-period change
    look_ahead_period = 5
    y = np.clip(np.abs(df['close'].pct_change(periods=look_ahead_period).shift(-look_ahead_period)) * 100, exit_remaining_percentage_from, exit_remaining_percentage_to)

    # Drop the last row from X and y to align their lengths
    X = X[:-look_ahead_period]
    y = y[:-look_ahead_period]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Optimized Gradient Boosting setup
    model = GradientBoostingRegressor(
        n_estimators=100,   # Increased for smoother predictions
        learning_rate=0.05, # Adjusted for better convergence
        max_depth=cpu_count,        # Balanced depth for available CPUs
        random_state=42
    )

    model.fit(X_train, y_train)

    # Save the trained model to disk
    joblib.dump(model, model_path)

    

# Update exit remaining percentage model
def update_exit_remaining_percentage_model(existing_model, df, exit_remaining_percentage_from, exit_remaining_percentage_to):
    """
    Update an existing GradientBoostingRegressor model with new data for exit remaining percentage prediction.
    
    Parameters:
        existing_model (GradientBoostingRegressor): The existing model to update.
        df (pd.DataFrame): The DataFrame containing the new data.
        exit_remaining_percentage_from (float): The lower bound for the exit remaining percentage.
        exit_remaining_percentage_to (float): The upper bound for the exit remaining percentage.
    
    Returns:
        GradientBoostingRegressor: The updated model.
    """
    df = calculate_indicators(df)  # Ensure technical indicators are calculated

    # Feature selection for the exit remaining percentage model
    X = df[['MACD', 'MACD_signal', 'EMA5', 'RSI', 'Momentum', 'ATR', 'Volume']].values

    # Adjusted target calculation with a look-ahead window for multi-period change
    look_ahead_period = 5
    y = np.clip(np.abs(df['close'].pct_change(periods=look_ahead_period).shift(-look_ahead_period)) * 100, exit_remaining_percentage_from, exit_remaining_percentage_to)

    # Drop the last row from X and y to align their lengths
    X = X[:-look_ahead_period]
    y = y[:-look_ahead_period]

    # Train-test split (optional, if you want to evaluate the updated model)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Get the number of CPU cores
    num_cores = cpu_count

    # Enable warm_start to add more trees
    existing_model.set_params(
        warm_start=True,  # Enable incremental training
        n_estimators=existing_model.n_estimators + 50,  # Add 50 more trees (adjust as needed)
        max_depth=num_cores,  # Set max_depth based on CPU cores (optional, but not recommended)
        random_state=42
    )

    # Retrain the model on the new data
    existing_model.fit(X_train, y_train)

# Train the machine learning models for all currency pairs and timeframes
def train_scalping_models(token, pair, timeframe, stop_loss_percentage, profit_target_from
    , profit_target_to, partial_exit_threshold_from, partial_exit_threshold_to
    , exit_remaining_percentage_from, exit_remaining_percentage_to, partial_exit_amount
):
    print("Training scalping models...")
    MODEL_KEY_TRAINED = f'Mockba/scalping_models/{pair}_{timeframe}_trading_model.joblib'
    local_model_path_training = f'temp/{pair}_{timeframe}_trading_model.joblib'

    MODEL_KEY_EXIT_REMAINING = f'Mockba/scalping_models/{pair}_{timeframe}_exit_remaining_percentage_model.joblib'
    local_model_path_exit_remaining = f'temp/{pair}_{timeframe}_exit_remaining_percentage_model.joblib'

    MODEL_KEY_PARCIAL_EXIT = f'Mockba/scalping_models/{pair}_{timeframe}_partial_exit_threshold_model.joblib'
    local_model_path_parcial_exit = f'temp/{pair}_{timeframe}_partial_exit_threshold_model.joblib'

    MODEL_KEY_PROFIT_TARGET = f'Mockba/scalping_models/{pair}_{timeframe}_profit_target_model.joblib'
    local_model_path_profit_target = f'temp/{pair}_{timeframe}_profit_target_model.joblib'

    # Get the current date
    now = datetime.now()
    current_date = now.strftime('%Y-%m-%d')
    values = f'2024-11-01|{current_date}'
    # Get the model path to verify if the model already exists
    # Fetch historical data and add technical indicators
    print("Fetch Historical Data")
    getHistorical.get_all_binance(pair, timeframe, token, save=True)
    # Get the historical data for the pair and timeframe
    print("Get Historical Data")
    data = get_historical_data(pair, timeframe, values)
    print("Add Indicators")
    data = calculate_indicators(data)
    print("Train Model")
    # Check if the model exists in DigitalOcean Spaces
    if download_model(BUCKET_NAME, MODEL_KEY_TRAINED, local_model_path_training):
        # Load the existing model
        print("Loaded existing model.")
        model = joblib.load(local_model_path_training)
        features = ['rsi', 'macd', 'macd_signal', 'volume', 'Momentum']
        update_model(model, data)
        upload_model(BUCKET_NAME, MODEL_KEY_TRAINED, local_model_path_training)
    else:
        # Train a new model if it doesn't exist
        print("No existing model found. Training a new model.")
        train_model(data, local_model_path_training)
        upload_model(BUCKET_NAME, MODEL_KEY_TRAINED, local_model_path_training)

    print("Train Model Exit remaining")
    # Check if the model exists in DigitalOcean Spaces
    if download_model(BUCKET_NAME, MODEL_KEY_EXIT_REMAINING, local_model_path_exit_remaining):
        # Load the existing model
        print("Loaded existing model.")
        model = joblib.load(local_model_path_exit_remaining)
        features = ['rsi', 'macd', 'macd_signal', 'volume', 'Momentum', 'ATR', 'EMA5']
        update_exit_remaining_percentage_model(model, data, exit_remaining_percentage_from, exit_remaining_percentage_to)
        upload_model(BUCKET_NAME, MODEL_KEY_EXIT_REMAINING, local_model_path_exit_remaining)
    else:
        # Train a new model if it doesn't exist
        print("No existing model found. Training a new model.")
        train_exit_remaining_percentage_model(data, local_model_path_exit_remaining, exit_remaining_percentage_from, exit_remaining_percentage_to)
        upload_model(BUCKET_NAME, MODEL_KEY_EXIT_REMAINING, local_model_path_exit_remaining)    

    print("Train Model Partial Exit")
    # Check if the model exists in DigitalOcean Spaces
    if download_model(BUCKET_NAME, MODEL_KEY_PARCIAL_EXIT, local_model_path_parcial_exit):
        # Load the existing model
        print("Loaded existing model.")
        model = joblib.load(local_model_path_parcial_exit)
        features = ['ATR', 'rsi']
        update_partial_exit_threshold_model(model, data, partial_exit_threshold_from, partial_exit_threshold_to)
        upload_model(BUCKET_NAME, MODEL_KEY_PARCIAL_EXIT, local_model_path_parcial_exit)
    else:
        # Train a new model if it doesn't exist
        print("No existing model found. Training a new model.")
        train_partial_exit_threshold_model(data, local_model_path_parcial_exit, partial_exit_threshold_from, partial_exit_threshold_to)
        upload_model(BUCKET_NAME, MODEL_KEY_PARCIAL_EXIT, local_model_path_parcial_exit) 

    print("Train Model Profit Target")
    # Check if the model exists in DigitalOcean Spaces
    if download_model(BUCKET_NAME, MODEL_KEY_PROFIT_TARGET, local_model_path_profit_target):
        # Load the existing model
        print("Loaded existing model.")
        model = joblib.load(local_model_path_profit_target)
        features = ['rsi', 'macd', 'macd_signal', 'Momentum', 'ATR', 'EMA5']
        update_profit_target_model(model, data, profit_target_from, profit_target_to)
        upload_model(BUCKET_NAME, MODEL_KEY_PROFIT_TARGET, local_model_path_profit_target)
    else:
        # Train a new model if it doesn't exist
        print("No existing model found. Training a new model.")
        train_profit_target_model(data, local_model_path_profit_target, profit_target_from, profit_target_to)
        upload_model(BUCKET_NAME, MODEL_KEY_PROFIT_TARGET, local_model_path_profit_target)            

    # Delete the local file after uploading
    if os.path.exists(local_model_path_training):
        os.remove(local_model_path_training)
        print(f"Deleted local file: {local_model_path_training}")
    else:
        print(f"Local file {local_model_path_training} does not exist.")

    # Delete the local file after uploading  
    if os.path.exists(local_model_path_exit_remaining):
        os.remove(local_model_path_exit_remaining)
        print(f"Deleted local file: {local_model_path_exit_remaining}")
    else:  
        print(f"Local file {local_model_path_exit_remaining} does not exist.")

    # Delete the local file after uploading
    if os.path.exists(local_model_path_parcial_exit):
        os.remove(local_model_path_parcial_exit)
        print(f"Deleted local file: {local_model_path_parcial_exit}")
    else:
        print(f"Local file {local_model_path_parcial_exit} does not exist.")

    # Delete the local file after uploading
    if os.path.exists(local_model_path_profit_target):
        os.remove(local_model_path_profit_target)
        print(f"Deleted local file: {local_model_path_profit_target}")
    else:
        print(f"Local file {local_model_path_profit_target} does not exist.")        

    print("Model training complete.")                  


if __name__ == "__main__":
    stop_loss_percentage = 0.5 # 50% stop loss
    profit_target_from = 0.1 # 1% profit target
    profit_target_to = 0.3 # 3% profit target
    partial_exit_threshold_from = 25.0 # 25% partial exit threshold
    partial_exit_threshold_to = 30.0 # 30% partial exit threshold
    exit_remaining_percentage_from = 15.0 # 15% exit remaining percentage
    exit_remaining_percentage_to = 20.0 # 20% exit remaining percentage
    partial_exit_amount = 0.15 # 15% partial exit amount

    train_scalping_models('000000', 'APTUSDT', '5m'
    , stop_loss_percentage, profit_target_from, profit_target_to
    , partial_exit_threshold_from, partial_exit_threshold_to
    , exit_remaining_percentage_from
    , exit_remaining_percentage_to, partial_exit_amount)