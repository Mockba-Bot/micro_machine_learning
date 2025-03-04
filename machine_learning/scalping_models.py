import sys
import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.utils.class_weight import compute_sample_weight
from scipy.stats import randint
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_sample_weight
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
    # Ensure numeric type
    data[['close', 'high', 'low', 'volume']] = data[['close', 'high', 'low', 'volume']].apply(pd.to_numeric)

    # --- Short-Term EMAs (Scalping) ---
    if 'ema_5' in required_features:
        data['ema_5'] = data['close'].ewm(span=5, adjust=False).mean()
    if 'ema_10' in required_features:
        data['ema_10'] = data['close'].ewm(span=10, adjust=False).mean()

    # --- RSI ---
    if 'rsi_14' in required_features:
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        data['rsi_14'] = 100 - (100 / (1 + rs))

    # --- Stochastic Oscillator ---
    if any(x in required_features for x in ['stoch_k', 'stoch_d']):
        data['stoch_k'] = ((data['close'] - data['low'].rolling(14).min()) /
                           (data['high'].rolling(14).max() - data['low'].rolling(14).min())) * 100
        data['stoch_d'] = data['stoch_k'].rolling(3).mean()

    # --- ATR (Short-Term for Scalping) ---
    if 'atr_14' in required_features:
        data['tr'] = pd.concat([
            data['high'] - data['low'],
            (data['high'] - data['close'].shift()).abs(),
            (data['low'] - data['close'].shift()).abs()
        ], axis=1).max(axis=1)
        data['atr_14'] = data['tr'].rolling(window=14).mean()

    # --- Bollinger Bands ---
    if any(x in required_features for x in ['bollinger_hband', 'bollinger_lband']):
        data['bollinger_mavg'] = data['close'].rolling(window=20).mean()
        data['bollinger_std'] = data['close'].rolling(window=20).std()
        data['bollinger_hband'] = data['bollinger_mavg'] + (data['bollinger_std'] * 2)
        data['bollinger_lband'] = data['bollinger_mavg'] - (data['bollinger_std'] * 2)

    # --- Volume-Based Features ---
    if 'volume_ma_10' in required_features:
        data['volume_ma_10'] = data['volume'].rolling(window=10).mean()
    
    if 'volume_delta' in required_features:
        data['volume_delta'] = data['volume'].diff()

    # --- Price Action Features ---
    if 'price_change_5' in required_features:
        data['price_change_5'] = data['close'].pct_change(periods=5) * 100  # Percentage change over 5 bars
    
    if 'high_low_diff' in required_features:
        data['high_low_diff'] = data['high'] - data['low']

    # Fill NaN values after calculations
    data.fillna(method='bfill', inplace=True)

    return data


# Train the machine learning model with advanced hyperparameter tuning
def train_model(data, model_path, features):
    """
    Train a Random Forest model with optimized hyperparameters for binary classification (Buy/Hold).
    """

    # --- 1️⃣ Create Labels (Target) with Upper & Lower Thresholds ---
    data['return'] = data['close'].pct_change().shift(-1)  # Predicting next period movement
    upper_threshold = data['return'].quantile(0.75)  # Top 25% → Buy
    lower_threshold = data['return'].quantile(0.25)  # Bottom 25% → Hold

    data['target'] = np.where(data['return'] > upper_threshold, 1, 0)

    # --- 2️⃣ Handle Missing Values ---
    data = data.dropna()

    # --- 3️⃣ Prepare Data ---
    X = data[features]
    y = data['target']

    # --- 4️⃣ Feature Selection (Mutual Information) ---
    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_df = pd.DataFrame({'Feature': X.columns, 'MI_Score': mi_scores})
    selected_features = mi_df[mi_df['MI_Score'] > 0.01]['Feature'].tolist()
    X = X[selected_features]

    # --- 5️⃣ Remove Highly Correlated Features ---
    correlation_matrix = X.corr().abs()
    upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    high_correlation_features = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.90)]
    X = X.drop(columns=high_correlation_features)

    # --- 6️⃣ Normalize Features ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- 7️⃣ Handle Class Imbalance (Resampling) ---
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    # --- 8️⃣ Compute Sample Weights *AFTER* Resampling ---
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_resampled)

    # --- 9️⃣ StratifiedKFold Cross-Validation ---
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # --- 🔟 Define Hyperparameter Space ---
    param_distributions = {
        'n_estimators': randint(500, 1500),  
        'max_depth': randint(20, 100),  
        'min_samples_split': randint(5, 20),  
        'min_samples_leaf': randint(2, 10),  
        'max_features': ['sqrt', 'log2'],
        'class_weight': ['balanced', 'balanced_subsample'],
    }

    randomized_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42), 
        param_distributions, 
        n_iter=100,
        cv=skf,  
        scoring='roc_auc',  
        random_state=42,
        n_jobs=cpu_count
    )

    randomized_search.fit(X_resampled, y_resampled, sample_weight=sample_weights)

    # --- 1️⃣1️⃣ Get the Best Model ---
    best_model = randomized_search.best_estimator_

    # --- 1️⃣2️⃣ Evaluate Performance ---
    y_pred = best_model.predict(X_resampled)
    y_pred_proba = best_model.predict_proba(X_resampled)[:, 1]

    print(f"Accuracy: {accuracy_score(y_resampled, y_pred):.4f}")
    print(f"Precision: {precision_score(y_resampled, y_pred):.4f}")
    print(f"Recall: {recall_score(y_resampled, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_resampled, y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_resampled, y_pred_proba):.4f}")

    # --- 1️⃣3️⃣ Save the Model ---
    joblib.dump(best_model, model_path)
    print(f"✅ Model trained and saved to {model_path}")


# Update the existing model with new data
def update_model(existing_model, new_data, features):
    """
    Update the existing Random Forest model with new data using the `warm_start` approach.
    """
    if not isinstance(existing_model, RandomForestClassifier):
        raise ValueError("Expected a RandomForestClassifier model for incremental training")

    # --- 1️⃣ Calculate Return & Define Target ---
    new_data['return'] = new_data['close'].pct_change().shift(-1)
    upper_threshold = new_data['return'].quantile(0.75)
    lower_threshold = new_data['return'].quantile(0.25)
    new_data['target'] = np.where(new_data['return'] > upper_threshold, 1, 0)  # Buy = 1, Hold = 0

    # --- 2️⃣ Handle Missing Values ---
    new_data = new_data.dropna()

    # --- 3️⃣ Prepare Dataset ---
    X_new = new_data[features]
    y_new = new_data['target']

    # --- 4️⃣ Feature Selection (Mutual Information) ---
    mi_scores = mutual_info_classif(X_new, y_new, random_state=42)
    mi_df = pd.DataFrame({'Feature': X_new.columns, 'MI_Score': mi_scores})
    selected_features = mi_df[mi_df['MI_Score'] > 0.01]['Feature'].tolist()
    X_new = X_new[selected_features]

    # --- 5️⃣ Normalize Features ---
    scaler = StandardScaler()
    X_new_scaled = scaler.fit_transform(X_new)

    # --- 6️⃣ Handle Class Imbalance (SMOTE) ---
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_new_scaled, y_new)

    # --- 7️⃣ Compute Sample Weights *After* Resampling ---
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_resampled)

    # --- 8️⃣ Update Model with New Data (Warm Start) ---
    existing_model.set_params(warm_start=True)  # Enable incremental learning
    existing_model.n_estimators += 50  # Add 50 new trees
    existing_model.fit(X_resampled, y_resampled, sample_weight=sample_weights)

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