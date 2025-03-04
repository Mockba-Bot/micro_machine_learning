import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
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
ddef add_indicators(data, required_features):
    # Ensure the columns are of numeric type
    data[['close', 'high', 'low', 'volume']] = data[['close', 'high', 'low', 'volume']].apply(pd.to_numeric)

    # --- Short-Term Moving Averages ---
    if 'ema_5' in required_features:
        data['ema_5'] = data['close'].ewm(span=5, adjust=False).mean()
    if 'ema_10' in required_features:
        data['ema_10'] = data['close'].ewm(span=10, adjust=False).mean()

    # --- Momentum Indicators ---
    if 'rsi_14' in required_features:
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        data['rsi_14'] = 100 - (100 / (1 + rs))

    if any(x in required_features for x in ['stoch_k', 'stoch_d']):
        data['stoch_k'] = ((data['close'] - data['low'].rolling(14).min()) /
                           (data['high'].rolling(14).max() - data['low'].rolling(14).min())) * 100
        data['stoch_d'] = data['stoch_k'].rolling(3).mean()

    # --- Volatility Indicators ---
    if 'atr_14' in required_features:
        data['tr'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - data['close'].shift()).abs(),
            (data['low'] - data['close'].shift()).abs()
        ], axis=1).max(axis=1)
        data['atr_14'] = data['tr'].rolling(window=14).mean()

    if any(x in required_features for x in ['bollinger_hband', 'bollinger_lband']):
        data['bollinger_mavg'] = data['close'].rolling(window=20).mean()
        data['bollinger_std'] = data['close'].rolling(window=20).std()
        data['bollinger_hband'] = data['bollinger_mavg'] + (data['bollinger_std'] * 2)
        data['bollinger_lband'] = data['bollinger_mavg'] - (data['bollinger_std'] * 2)

    # --- Volume-Based Indicators ---
    if 'volume_ma_10' in required_features:
        data['volume_ma_10'] = data['volume'].rolling(window=10).mean()
    if 'volume_delta' in required_features:
        data['volume_delta'] = data['volume'].diff()

    # --- Price Action Features ---
    if 'price_change_5' in required_features:
        data['price_change_5'] = data['close'].pct_change(periods=5) * 100
    if 'high_low_diff' in required_features:
        data['high_low_diff'] = data['high'] - data['low']

    # Fill NaN values after calculations
    data.fillna(method='bfill', inplace=True)

    return data


# Train the machine learning model with advanced hyperparameter tuning
def train_model(data, model_path, features):
    # Calculate return and target columns
    data['return'] = data['close'].pct_change().shift(1)  # Avoid look-ahead bias
    data['target'] = (data['return'] > 0).astype(int)  # Binary classification target
    
    # Handle missing values
    data = data.dropna()
    
    # Prepare training and testing datasets
    X = data[features]
    y = data['target']
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
        cv=5,  # Increased number of cross-validation folds
        scoring='roc_auc',  # Use ROC-AUC for imbalanced datasets
        random_state=42,
        n_jobs=cpu_count  # Use all available CPU cores
    )
    randomized_search.fit(X_train, y_train)
    
    # Get the best model from the search
    best_model = randomized_search.best_estimator_

    # Evaluate the model
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]  # Probabilities for ROC-AUC

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    
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
    MODEL_KEY = f'Mockba/scalping_models/scalping_model_{pair}_{timeframe}_{model}.pkl'
    local_model_path = f'temp/scalping_model_{pair}_{timeframe}_{model}.pkl'

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


if __name__ == "__main__":
    scalping_features = [
        "ema_5", "ema_10",  # Short-term moving averages
        "rsi_14", "stoch_k", "stoch_d",  # Momentum indicators
        "atr_14", "bollinger_hband", "bollinger_lband",  # Volatility indicators
        "volume_ma_10", "volume_delta",  # Volume-based indicators
        "price_change_5", "high_low_diff"  # Price action features
    ]
    intervals = ["5m"]

    # Iterate over each set of features and train models
    for i, feature_set in enumerate(scalping_features):
        print(f"Training models with feature set {i}: {feature_set}")
        train_models('PERP_APT_USDC', intervals, feature_set)