import sys
import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, make_scorer
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.feature_selection import mutual_info_classif
from sklearn.utils.class_weight import compute_sample_weight
from scipy.stats import randint
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_sample_weight
from dotenv import load_dotenv
import numpy as np
import joblib  # Library for model serialization
from datetime import datetime  # Import timedelta from datetime
# Add the directory containing your modules to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from historical_data import get_historical_data
import time
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

base_features = ["close", "high", "low", "volume"]

strategy_features = {
    "5m": {
        "Trend-Following": {"features": base_features + ["ema_12", "ema_26", "macd", "macd_signal", "adx", "vwap"], "force_features": True},
        "Volatility Breakout": {"features": base_features + ["atr_14", "bollinger_hband", "bollinger_lband", "std_20", "vwap"], "force_features": True},
        "Momentum Reversal": {"features": base_features + ["rsi_14", "stoch_k_14", "stoch_d_14", "roc_10", "momentum_10", "vwap"], "force_features": True},
        "Momentum + Volatility": {"features": base_features + ["rsi_14", "atr_14", "bollinger_hband", "bollinger_lband", "roc_10", "momentum_10", "vwap"], "force_features": True},
        "Hybrid": {"features": base_features + ["ema_12", "ema_26", "atr_14", "bollinger_hband", "rsi_14", "macd", "vwap"], "force_features": True},
        "Advanced": {"features": base_features + ["tenkan_sen_9", "kijun_sen_26", "senkou_span_a", "senkou_span_b", "sar", "vwap"], "force_features": True},
        "Router": {"features": base_features + ["ema_12", "ema_26", "macd", "macd_signal", "adx", "atr_14", "bollinger_hband", "bollinger_lband", "std_20", "rsi_14", "stoch_k_14", "stoch_d_14", "roc_10", "momentum_10", "tenkan_sen_9", "kijun_sen_26", "senkou_span_a", "senkou_span_b", "sar", "vwap"], "force_features": False}
    },
    "1h": {
        "Trend-Following": {"features": base_features + ["ema_20", "ema_50", "macd", "macd_signal", "adx", "vwap"], "force_features": True},
        "Volatility Breakout": {"features": base_features + ["atr_14", "bollinger_hband", "bollinger_lband", "std_20", "vwap"], "force_features": True},
        "Momentum Reversal": {"features": base_features + ["rsi_14", "stoch_k_14", "stoch_d_14", "roc_10", "momentum_10", "vwap"], "force_features": True},
        "Momentum + Volatility": {"features": base_features + ["rsi_14", "atr_14", "bollinger_hband", "bollinger_lband", "roc_10", "momentum_10", "vwap"], "force_features": True},
        "Hybrid": {"features": base_features + ["ema_20", "ema_50", "atr_14", "bollinger_hband", "rsi_14", "macd", "vwap"], "force_features": True},
        "Advanced": {"features": base_features + ["tenkan_sen_9", "kijun_sen_26", "senkou_span_a", "senkou_span_b", "sar", "vwap"], "force_features": True},
        "Router": {"features": base_features + ["ema_12", "ema_26", "macd", "macd_signal", "adx", "atr_14", "bollinger_hband", "bollinger_lband", "std_20", "rsi_14", "stoch_k_14", "stoch_d_14", "roc_10", "momentum_10", "tenkan_sen_9", "kijun_sen_26", "senkou_span_a", "senkou_span_b", "sar", "vwap"], "force_features": False}
    },
    "4h": {
        "Trend-Following": {"features": base_features + ["ema_50", "ema_200", "macd", "macd_signal", "adx", "vwap"], "force_features": True},
        "Volatility Breakout": {"features": base_features + ["atr_14", "bollinger_hband", "bollinger_lband", "std_20", "vwap"], "force_features": True},
        "Momentum Reversal": {"features": base_features + ["rsi_14", "stoch_k_14", "stoch_d_14", "roc_10", "momentum_10", "vwap"], "force_features": True},
        "Momentum + Volatility": {"features": base_features + ["rsi_14", "atr_14", "bollinger_hband", "bollinger_lband", "roc_10", "momentum_10", "vwap"], "force_features": True},
        "Hybrid": {"features": base_features + ["ema_50", "ema_200", "atr_14", "bollinger_hband", "rsi_14", "macd", "vwap"], "force_features": True},
        "Advanced": {"features": base_features + ["tenkan_sen_9", "kijun_sen_26", "senkou_span_a", "senkou_span_b", "sar", "vwap"], "force_features": True},
        "Router": {"features": base_features + ["ema_12", "ema_26", "macd", "macd_signal", "adx", "atr_14", "bollinger_hband", "bollinger_lband", "std_20", "rsi_14", "stoch_k_14", "stoch_d_14", "roc_10", "momentum_10", "tenkan_sen_9", "kijun_sen_26", "senkou_span_a", "senkou_span_b", "sar", "vwap"], "force_features": False}
    },
    "1d": {
        "Trend-Following": {"features": base_features + ["ema_50", "ema_200", "macd", "macd_signal", "adx", "vwap"], "force_features": True},
        "Volatility Breakout": {"features": base_features + ["atr_14", "bollinger_hband", "bollinger_lband", "std_20", "vwap"], "force_features": True},
        "Momentum Reversal": {"features": base_features + ["rsi_14", "stoch_k_14", "stoch_d_14", "roc_10", "momentum_10", "vwap"], "force_features": True},
        "Momentum + Volatility": {"features": base_features + ["rsi_14", "atr_14", "bollinger_hband", "bollinger_lband", "roc_10", "momentum_10", "vwap"], "force_features": True},
        "Hybrid": {"features": base_features + ["ema_50", "ema_200", "atr_14", "bollinger_hband", "rsi_14", "macd", "vwap"], "force_features": True},
        "Advanced": {"features": base_features + ["tenkan_sen_9", "kijun_sen_26", "senkou_span_a", "senkou_span_b", "sar", "vwap"], "force_features": True},
        "Router": {"features": base_features + ["ema_12", "ema_26", "macd", "macd_signal", "adx", "atr_14", "bollinger_hband", "bollinger_lband", "std_20", "rsi_14", "stoch_k_14", "stoch_d_14", "roc_10", "momentum_10", "tenkan_sen_9", "kijun_sen_26", "senkou_span_a", "senkou_span_b", "sar", "vwap"], "force_features": False}
    }
}


# Add technical indicators to the data
def add_indicators(data, required_features):
    """
    Add only the necessary indicators to the data based on the requested features.
    """
    # Ensure numeric columns
    data[['close', 'high', 'low', 'volume']] = data[['close', 'high', 'low', 'volume']].apply(pd.to_numeric)

    # --- EMA ---
    for feature in required_features:
        if feature.startswith("ema_"):
            try:
                window = int(feature.split("_")[1])  # Extract window size from feature name
                data[feature] = data['close'].ewm(span=window, adjust=False).mean()
            except (IndexError, ValueError):
                print(f"⚠️ Warning: Could not extract window size from feature: {feature}")

    # --- MACD ---
    if any(x in required_features for x in ['macd', 'macd_signal']):
        data['ema_12'] = data['close'].ewm(span=12, adjust=False).mean()
        data['ema_26'] = data['close'].ewm(span=26, adjust=False).mean()
        data['macd'] = data['ema_12'] - data['ema_26']
        data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()

    # --- ATR ---
    for feature in required_features:
        if feature.startswith("atr_"):
            try:
                window = int(feature.split("_")[1])  # Extract window size from feature name
                data['tr'] = pd.concat([
                    data['high'] - data['low'],
                    (data['high'] - data['close'].shift()).abs(),
                    (data['low'] - data['close'].shift()).abs()
                ], axis=1).max(axis=1)
                data[feature] = data['tr'].rolling(window=window).mean()
            except (IndexError, ValueError):
                print(f"⚠️ Warning: Could not extract window size from feature: {feature}")

    # --- Bollinger Bands ---
    for feature in required_features:
        if feature.startswith("bollinger_"):
            try:
                window = int(feature.split("_")[-1])  # Extract window size from feature name
                data['bollinger_mavg'] = data['close'].rolling(window=window).mean()
                data['bollinger_std'] = data['close'].rolling(window=window).std()
                data['bollinger_hband'] = data['bollinger_mavg'] + (data['bollinger_std'] * 2)
                data['bollinger_lband'] = data['bollinger_mavg'] - (data['bollinger_std'] * 2)
            except (IndexError, ValueError):
                print(f"⚠️ Warning: Could not extract window size from feature: {feature}")

    # --- Standard Deviation ---
    for feature in required_features:
        if feature.startswith("std_"):
            try:
                window = int(feature.split("_")[1])  # Extract window size from feature name
                data[feature] = data['close'].rolling(window=window).std()
            except (IndexError, ValueError):
                print(f"⚠️ Warning: Could not extract window size from feature: {feature}")

    # --- RSI ---
    for feature in required_features:
        if feature.startswith("rsi_"):
            try:
                window = int(feature.split("_")[1])  # Extract window size from feature name
                delta = data['close'].diff()
                gain = delta.where(delta > 0, 0).rolling(window=window).mean()
                loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
                rs = gain / loss
                data[feature] = 100 - (100 / (1 + rs))
            except (IndexError, ValueError):
                print(f"⚠️ Warning: Could not extract window size from feature: {feature}")

    # --- Stochastic Oscillator ---
    for feature in required_features:
        if feature.startswith("stoch_"):
            try:
                window = int(feature.split("_")[-1])  # Extract window size from feature name
                data['stoch_k'] = ((data['close'] - data['low'].rolling(window).min()) /
                                   (data['high'].rolling(window).max() - data['low'].rolling(window).min())) * 100
                data['stoch_d'] = data['stoch_k'].rolling(3).mean()
            except (IndexError, ValueError):
                print(f"⚠️ Warning: Could not extract window size from feature: {feature}")

    # --- Momentum ---
    for feature in required_features:
        if feature.startswith("momentum_"):
            try:
                window = int(feature.split("_")[1])  # Extract window size from feature name
                data[feature] = data['close'].diff(periods=window)
            except (IndexError, ValueError):
                print(f"⚠️ Warning: Could not extract window size from feature: {feature}")

    # --- Rate of Change (ROC) ---
    for feature in required_features:
        if feature.startswith("roc_"):
            try:
                window = int(feature.split("_")[1])  # Extract window size from feature name
                data[feature] = data['close'].pct_change(periods=window) * 100
            except (IndexError, ValueError):
                print(f"⚠️ Warning: Could not extract window size from feature: {feature}")

     
        # --- ADX ---
    for feature in required_features:
        if feature.startswith("adx"):
            try:
                # If the feature has an underscore, assume the portion after it is the window size
                if "_" in feature:
                    window = int(feature.split("_")[1])
                else:
                    window = 14  # Default window for ADX if no underscore

                data['plus_dm'] = data['high'].diff().where(lambda x: x > 0, 0)
                data['minus_dm'] = -data['low'].diff().where(lambda x: x < 0, 0)

                # Calculate True Range (TR)
                data['tr'] = pd.concat([
                    data['high'] - data['low'],
                    (data['high'] - data['close'].shift()).abs(),
                    (data['low'] - data['close'].shift()).abs()
                ], axis=1).max(axis=1)

                # Calculate +DI and -DI
                data['plus_di'] = 100 * (
                    data['plus_dm'].rolling(window=window).mean()
                    / data['tr'].rolling(window=window).mean()
                )
                data['minus_di'] = 100 * (
                    data['minus_dm'].rolling(window=window).mean()
                    / data['tr'].rolling(window=window).mean()
                )

                # Calculate ADX
                data['dx'] = 100 * abs(data['plus_di'] - data['minus_di']) / (data['plus_di'] + data['minus_di'])
                data[feature] = data['dx'].rolling(window=window).mean()

            except (IndexError, ValueError) as e:
                print(f"⚠️ Warning: Could not extract window size from feature: {feature}. Error: {e}")

    # --- Ichimoku Cloud ---
    for feature in required_features:
        if feature.startswith("tenkan_sen_"):
            try:
                window = int(feature.split("_")[-1])  # Extract window size from feature name
                data[feature] = (data['high'].rolling(window=window).max() + data['low'].rolling(window=window).min()) / 2
            except (IndexError, ValueError):
                print(f"⚠️ Warning: Could not extract window size from feature: {feature}")
        if feature.startswith("kijun_sen_"):
            try:
                window = int(feature.split("_")[-1])  # Extract window size from feature name
                data[feature] = (data['high'].rolling(window=window).max() + data['low'].rolling(window=window).min()) / 2
            except (IndexError, ValueError):
                print(f"⚠️ Warning: Could not extract window size from feature: {feature}")
        if feature.startswith("senkou_span_a"):
            data[feature] = ((data['tenkan_sen_9'] + data['kijun_sen_26']) / 2).shift(26)
        if feature.startswith("senkou_span_b"):
            data[feature] = ((data['high'].rolling(window=52).max() + data['low'].rolling(window=52).min()) / 2).shift(26)

    # --- Parabolic SAR ---
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

    # --- VWAP ---
    if 'vwap' in required_features:
        data['vwap'] = (data['volume'] * (data['high'] + data['low'] + data['close']) / 3).cumsum() / data['volume'].cumsum()

    # Fill NaN values after calculations
    data.fillna(method='bfill', inplace=True)

    return data


def safe_roc_auc_score(y_true, y_pred_proba, sample_weight=None, **kwargs):
    all_classes = [-1, 0, 1]
    unique_test_labels = np.unique(y_true)

    if len(unique_test_labels) < 2:
        return np.nan  # ROC-AUC is undefined for fewer than 2 classes

    # Ensure y_pred_proba is 2D
    if len(y_pred_proba.shape) == 1:
        y_pred_proba = y_pred_proba.reshape(-1, 1)  # Reshape to (n_samples, 1)

    if len(unique_test_labels) == 2:
        # Binary classification
        pos_label = sorted(unique_test_labels)[1]
        pos_index = all_classes.index(pos_label)
        return roc_auc_score(
            y_true, 
            y_pred_proba[:, pos_index], 
            sample_weight=sample_weight
        )
    else:
        # Multi-class classification
        y_true_onehot = label_binarize(y_true, classes=all_classes)
        return roc_auc_score(
            y_true_onehot,
            y_pred_proba,
            multi_class='ovr',
            average='macro',
            sample_weight=sample_weight
        )

# Tune the Mutual Information threshold
def tune_mi_threshold(X, y, thresholds=[0.005, 0.01, 0.02, 0.05, 0.08]):
    """
    Automatically finds the best mutual information (MI) threshold.
    """
    best_threshold = None
    best_score = -np.inf
    best_features = None

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for threshold in thresholds:
        mi_scores = mutual_info_classif(X, y, random_state=42)
        mi_df = pd.DataFrame({'Feature': X.columns, 'MI_Score': mi_scores})
        selected_features = mi_df[mi_df['MI_Score'] > threshold]['Feature'].tolist()

        if len(selected_features) < 2:
            # Skip if too few features remain
            continue

        X_selected = X[selected_features]

        # Normalize Features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_selected)

        # Handle Class Imbalance
        smote = SMOTE(sampling_strategy='auto', random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

        # Train a basic model using cross-validation
        rf = RandomForestClassifier(random_state=42, n_estimators=500)
        scores = []
        for train_idx, test_idx in skf.split(X_resampled, y_resampled):
            X_train, X_test = X_resampled[train_idx], X_resampled[test_idx]
            y_train, y_test = y_resampled[train_idx], y_resampled[test_idx]

            # Skip if fewer than 2 classes in y_test
            if len(np.unique(y_test)) < 2:
                scores.append(np.nan)
                continue

            rf.fit(X_train, y_train)
            y_pred_proba = rf.predict_proba(X_test)  # Probability estimates

            # Ensure y_pred_proba is 2D
            if len(y_pred_proba.shape) == 1:
                y_pred_proba = y_pred_proba.reshape(-1, 1)

            # Calculate ROC-AUC
            roc_auc = safe_roc_auc_score(y_test, y_pred_proba)
            scores.append(roc_auc)

        mean_score = np.nanmean(scores)  # Safely handle any np.nan
        # print(f"Threshold {threshold}: Selected {len(selected_features)} features - ROC-AUC: {mean_score:.4f}")

        # Store the best threshold if improvement
        if mean_score > best_score:
            best_score = mean_score
            best_threshold = threshold
            best_features = selected_features

    # Print the best threshold and selected features
    # print(f"✅ Best MI Threshold: {best_threshold}, Features Selected: {len(best_features)}")
    # print(f"✅ Selected Features: {best_features}")

    return best_threshold, best_features


# Train the model with optimized hyperparameters and automatic MI threshold tuning
def train_model(data, model_path, interval, strategy, BUCKET_NAME, MODEL_KEY, force_features=False):
    """
    Train a Random Forest model with features selected based on the interval and strategy.
    Ensure that 'close' is always included in the final feature set.
    """
    # Get features for the specified interval and strategy
    features_dict = get_features_for_strategy(interval, strategy)
    features = features_dict["features"]
    if not features:
        raise ValueError(f"No features defined for interval: {interval} and strategy: {strategy}")

    # Ensure essential market data is included
    base_features = ['close', 'high', 'low', 'volume']
    full_features = list(set(base_features + features))  # Merge required & requested features

    # Create Labels (Target) for Buy (1), Hold (0), Sell (-1)
    data['return'] = data['close'].pct_change().shift(-1)  # Predicting next period movement
    upper_threshold = data['return'].quantile(0.85)  # Buy threshold
    lower_threshold = data['return'].quantile(0.15)  # Sell threshold

    # 🛠 Assign Labels:
    #  1 = Buy (future return is high)
    #  0 = Hold (neutral range)
    # -1 = Sell (future return is low)
    data['target'] = np.where(data['return'] > upper_threshold, 1, 
                      np.where(data['return'] < lower_threshold, -1, 0))
    
    # Print how many signals 1, 0, and -1 exist
    # print("Signal distribution:")
    # print(data['target'].value_counts())

    # Handle missing values
    data = data.dropna()

    # Prepare dataset
    X = data[full_features]
    y = data['target'].values  # Ensure y is a 1D array

    # Debugging: Print the shape of y
    # print(f"Shape of y: {y.shape}")  # Should output (n_samples,)

    # If y is not 1D, flatten it
    if len(y.shape) > 1:
        y = y.ravel()  # Flatten to 1D

    # **Auto-tune MI threshold**
    if force_features:
        selected_features = list(X.columns)  # Use all features directly
        print(f"🚨 Skipping MI thresholding. Using all {len(selected_features)} strategy features.")
    else:
        best_threshold, selected_features = tune_mi_threshold(X, y)


    # Use the best-selected features
    X = X[selected_features]

    # Remove highly correlated features
    correlation_matrix = X.corr().abs()
    # print("Correlation Matrix:")
    # print(correlation_matrix)

    # Calculate dynamic correlation threshold
    num_features = len(selected_features)
    min_features_to_retain = int(0.85 * num_features)  # Retain at least 75% of features
    # print(f"Total features: {num_features}, Min features to retain: {min_features_to_retain}")

    if force_features:
        final_features = selected_features
        # print(f"🚨 Skipping correlation filtering. Using all {len(final_features)} strategy features (forced).")
    else:
        # Remove highly correlated features
        upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
        high_correlation_features = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.98)]  # Adjusted threshold
        final_features = [f for f in selected_features if f not in high_correlation_features]

    # Ensure 'close' is always included in the final features
    if 'close' not in final_features:
        print("⚠️ Warning: 'close' was not in the final features. Adding it back.")
        final_features.append('close')

    # Ensure at least 75% of features are retained
    if len(final_features) < min_features_to_retain:
        print(f"⚠️ Warning: Only {len(final_features)} features remain after removing correlations. Retaining top {min_features_to_retain} features.")
        final_features = selected_features[:min_features_to_retain]  # Retain top 75% features

    # Ensure 'close' is still present after retaining top features
    if 'close' not in final_features:
        print("⚠️ Warning: 'close' was not in the retained features. Adding it back.")
        final_features.append('close')

    X = X[final_features]

    # Normalize Features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Handle Class Imbalance
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    # Compute Sample Weights
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_resampled)

    # Hyperparameter tuning with RandomizedSearchCV
    param_distributions = {
        'n_estimators': randint(500, 1500),  
        'max_depth': randint(20, 100),  
        'min_samples_split': randint(5, 20),  
        'min_samples_leaf': randint(2, 10),  
        'max_features': ['sqrt', 'log2'],
        'class_weight': ['balanced', 'balanced_subsample'],
    }

    # Create a custom multi-class ROC-AUC scorer
    custom_auc_scorer = make_scorer(safe_roc_auc_score, needs_proba=True)

    randomized_search = RandomizedSearchCV( 
        RandomForestClassifier(random_state=42),
        param_distributions=param_distributions,
        n_iter=200,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring=custom_auc_scorer,
        random_state=42,
        n_jobs=cpu_count
    )

    randomized_search.fit(X_resampled, y_resampled, sample_weight=sample_weights)

    # Get the best model
    best_model = randomized_search.best_estimator_

    # Evaluate Performance
    y_pred = best_model.predict(X_resampled)
    y_pred_proba = best_model.predict_proba(X_resampled)

    # Debugging: Check the shape of y_pred_proba
    # print(f"Shape of y_pred_proba: {y_pred_proba.shape}")  # Should be (n_samples, n_classes)

    # Ensure y_pred_proba is 2D
    if len(y_pred_proba.shape) == 1:
        y_pred_proba = y_pred_proba.reshape(-1, 1)  # Reshape to (n_samples, 1)

    print(f"✅ Accuracy: {accuracy_score(y_resampled, y_pred):.4f}")
    print(f"✅ Precision: {precision_score(y_resampled, y_pred, average='macro'):.4f}")
    print(f"✅ Recall: {recall_score(y_resampled, y_pred, average='macro'):.4f}")
    print(f"✅ F1-Score: {f1_score(y_resampled, y_pred, average='macro'):.4f}")
    print(f"✅ ROC-AUC: {roc_auc_score(y_resampled, y_pred_proba, multi_class='ovr'):.4f}")

    # Save Model & Features
    model_metadata = {
        "model": best_model,
        "used_features": final_features  # Save the final features used for training
    }
    joblib.dump(model_metadata, model_path, compress=3)
    # print(f"✅ Model trained and saved at {model_path} with features: {final_features}")
    upload_model(BUCKET_NAME, MODEL_KEY, model_path)
    # print(f"✅ Model uploaded to {BUCKET_NAME}/{MODEL_KEY}")


# Update the existing model with new data
def update_model(model_path, new_data, BUCKET_NAME, MODEL_KEY):
    """
    Update an existing Random Forest model with new data using incremental learning (warm start).
    Retains the same feature set and preprocessing steps as the original model.
    """
    # --- 1️⃣ Load and Extract the Model ---
    try:
        model_metadata = joblib.load(model_path)

        if not isinstance(model_metadata, dict) or "model" not in model_metadata:
            raise ValueError("Invalid model format. Expected a dictionary with 'model' key.")
        
        existing_model = model_metadata["model"]
        trained_features = model_metadata["used_features"]  # Use the same features as the original model

        # Ensure the extracted model is a RandomForestClassifier
        if not isinstance(existing_model, RandomForestClassifier):
            raise ValueError("Expected a RandomForestClassifier model for incremental training.")

    except FileNotFoundError:
        raise ValueError(f"Model file {model_path} not found. Ensure the model is trained first.")

    # --- 2️⃣ Calculate Return & Define Target ---
    new_data['return'] = new_data['close'].pct_change().shift(-1)  # Predicting next period movement
    upper_threshold = new_data['return'].quantile(0.85)  # Buy threshold
    lower_threshold = new_data['return'].quantile(0.15)  # Sell threshold

    # 🛠 Assign Labels:
    #  1 = Buy (future return is high)
    #  0 = Hold (neutral range)
    # -1 = Sell (future return is low)
    new_data['target'] = np.where(new_data['return'] > upper_threshold, 1, 
                          np.where(new_data['return'] < lower_threshold, -1, 0))
    
    # Print how many signals 1, 0, and -1 exist
    # print("Signal distribution:")
    # print(new_data['target'].value_counts())

    # --- 3️⃣ Handle Missing Values ---
    new_data = new_data.dropna()

    # ✅ Add technical indicators if missing
    new_data = add_indicators(new_data, trained_features)

    # --- 4️⃣ Prepare Dataset ---
    X_new = new_data[trained_features]  # Use the same features as the original model
    y_new = new_data['target'].values  # Ensure y is a 1D array

    # If y is not 1D, flatten it
    if len(y_new.shape) > 1:
        y_new = y_new.ravel()  # Flatten to 1D

    # --- 5️⃣ Normalize Features ---
    scaler = StandardScaler()
    X_new_scaled = scaler.fit_transform(X_new)

    # --- 6️⃣ Handle Class Imbalance (SMOTE) ---
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_new_scaled, y_new)

    # --- 7️⃣ Compute Sample Weights *After* Resampling ---
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_resampled)

    # --- 8️⃣ Incremental Learning (Warm Start) ---
    existing_model.set_params(warm_start=True)  # Enable incremental training
    new_trees = max(50, int(0.2 * existing_model.n_estimators))
    # Add new trees to the existing model
    # Note: n_estimators should be increased by the number of new trees
    # to prevent overfitting on the new data
    # The new trees should be at least 50 or 20% of the existing trees
    # to ensure the model adapts to the new data
    existing_model.n_estimators += new_trees
    existing_model.fit(X_resampled, y_resampled, sample_weight=sample_weights)

    # --- 9️⃣ Evaluate Updated Model ---
    y_pred = existing_model.predict(X_resampled)
    y_pred_proba = existing_model.predict_proba(X_resampled)

    # Debugging: Check the shape of y_pred_proba
    print(f"Shape of y_pred_proba: {y_pred_proba.shape}")  # Should be (n_samples, n_classes)

    # Ensure y_pred_proba is 2D
    if len(y_pred_proba.shape) == 1:
        y_pred_proba = y_pred_proba.reshape(-1, 1)  # Reshape to (n_samples, 1)

    print(f"✅ Accuracy: {accuracy_score(y_resampled, y_pred):.4f}")
    print(f"✅ Precision: {precision_score(y_resampled, y_pred, average='macro'):.4f}")
    print(f"✅ Recall: {recall_score(y_resampled, y_pred, average='macro'):.4f}")
    print(f"✅ F1-Score: {f1_score(y_resampled, y_pred, average='macro'):.4f}")
    print(f"✅ ROC-AUC: {roc_auc_score(y_resampled, y_pred_proba, multi_class='ovr'):.4f}")

    # --- 🔟 Save Updated Model ---
    model_metadata = {
        "model": existing_model,
        "used_features": trained_features  # Retain the same features as the original model
    }
    joblib.dump(model_metadata, model_path, compress=3)
    # print(f"✅ Model updated and saved to {model_path}")
    upload_model(BUCKET_NAME, MODEL_KEY, model_path)
    # print(f"✅ Model uploaded to {BUCKET_NAME}/{MODEL_KEY}")

    return existing_model

# Train the machine learning model with advanced hyperparameter tuning
def train_machine_learning(pair, interval, strategy):
 
    features_dict = get_features_for_strategy(interval, strategy)
    features = features_dict["features"]
    force_features = features_dict["force_features"]
    if not features:
        raise ValueError(f"No features defined for interval: {interval} and strategy: {strategy}")

    model = "_".join(features).replace("[", "").replace("]", "").replace("'", "_").replace(" ", "")
    MODEL_KEY = f'Mockba/trained_models/trained_model_{pair}_{interval}_{model}.joblib'
    local_model_path = f'temp/trained_model_{pair}_{interval}_{model}.joblib'

    # Get the current date
    now = datetime.now()
    current_date = now.strftime('%Y-%m-%d')
    values = f'2024-01-01|{current_date}'

    # Get historical data
    data = get_historical_data(pair, interval, values)

    # Add technical indicators
    data = add_indicators(data, features)

    # Automatically determine the feature columns (Exclude non-numeric ones)
    exclude_columns = ['start_timestamp']
    features = [col for col in data.columns if col not in exclude_columns]

    # Check if the model exists in storage
    if download_model(BUCKET_NAME, MODEL_KEY, local_model_path):
        # Load existing model
        # print("Loaded existing model.")
        update_model(local_model_path, data, BUCKET_NAME, MODEL_KEY)
    else:
        # Train a new model if none exists
        # print("No existing model found. Training a new model.")
        train_model(data, local_model_path, interval, strategy, BUCKET_NAME, MODEL_KEY, force_features)

    # Delete local model file after upload
    if os.path.exists(local_model_path):
        os.remove(local_model_path)
    else:
        print(f"Local file {local_model_path} does not exist.")

    print("✅ Model training complete.")    


def get_features_for_strategy(interval, strategy):
    """
    Returns the interval, strategy, and the list of features for the given interval and strategy.
    
    Parameters:
        interval (str): The timeframe interval (e.g., '5m', '1h', '4h', '1d').
        strategy (str): The strategy name (e.g., 'Trend-Following', 'Volatility Breakout').
    
    Returns:
        dict: A dictionary with 'interval', 'strategy', 'features', and 'force_features' keys.
    """
    strategy_info = strategy_features.get(interval, {}).get(strategy, {})
    return {
        "interval": interval,
        "strategy": strategy,
        "features": strategy_info.get("features", []),
        "force_features": strategy_info.get("force_features", False)
    }


# Main function to train or update models for multiple intervals
def train_models(symbol, intervals, strategies):
    for interval in intervals:
        for strategy in strategies:
            train_machine_learning(symbol, interval, strategy)
        

# Train models with different features
# if __name__ == "__main__":
#     # User selects interval and strategy
#     interval = "1h"
#     strategy = "Trend-Following"
#     train_machine_learning('PERP_APT_USDC', interval, strategy)