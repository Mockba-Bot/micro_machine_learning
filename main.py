import sys
import os
from dotenv import load_dotenv
import redis
import requests
from datetime import timedelta
import time  # Import time module
import schedule

# Add the directory containing your modules to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'machine_learning')))
from machine_learning import bucket, elliot_waves_analysis, scalping_models, signal_models, technical_analysis, training_models
from database.OrderlyKlines import run_all_timeframes

# Load environment variables from the .env file
load_dotenv(dotenv_path=".env.micro.machine.learning")

# ✅ Orderly API Config
BASE_URL = os.getenv("ORDERLY_BASE_URL")

# ✅ Fetch Orderly Trading Pairs
def fetch_orderly_symbols():
    url = f"{BASE_URL}/v1/public/info"
    try:
        response = requests.get(url)
        data = response.json()
        if data.get("success") and "data" in data:
            symbols = [row["symbol"] for row in data["data"]["rows"] if "symbol" in row]
            return symbols
        else:
            print("⚠️ Unexpected API response format or missing 'data' key.")
            return []
    except Exception as e:
        print(f"❌ Error fetching Orderly symbols: {e}")
        return []

def run_machine_learning_and_historical_data():
    symbols = fetch_orderly_symbols()
    #run historical data first
    run_all_timeframes(symbols)
    #run machine learning
    for symbol in symbols:
        # if symbol == 'PERP_APT_USDC':
        intervals = ['1h', '4h', '1d']
        try:
            # Training for Elliot Waves
            print("---TRAINING ELLIOT WAVES---")
            elliot_waves_analysis.train_models(symbol, intervals)
            time.sleep(2)

            print("---TRAINING SCALPING MODELS---")
            # Training for scalping models
            stop_loss_percentage = 10 # 10% stop loss
            profit_target_from = 0.1 # 1% profit target
            profit_target_to = 0.3 # 3% profit target
            partial_exit_threshold_from = 25.0 # 25% partial exit threshold
            partial_exit_threshold_to = 30.0 # 30% partial exit threshold
            exit_remaining_percentage_from = 15.0 # 15% exit remaining percentage
            exit_remaining_percentage_to = 20.0 # 20% exit remaining percentage
            partial_exit_amount = 0.15 # 15% partial exit amount
            scalping_models.train_scalping_models('000000', symbol, '5m', stop_loss_percentage, profit_target_from, profit_target_to, partial_exit_threshold_from, partial_exit_threshold_to, exit_remaining_percentage_from, exit_remaining_percentage_to, partial_exit_amount)
            time.sleep(2)

            print("---TRAINING SIGNAL MODELS---")
            # Training for signal models
            signal_models.train_models(symbol, intervals)
            time.sleep(2)

            print("---TRAINING TECHNICAL ANALYSIS---")
            # # Training for technical analysis
            technical_analysis.train_models(symbol, intervals)
            time.sleep(2)

            print("---TRAINING MACHINE LEARNING MODELS---")
            # Training for machine learning models
            # Define the features array
            features = [
                ["rsi", "macd", "macd_signal", "macd_diff", "bollinger_hband", "bollinger_mavg", "bollinger_lband", "ema_20", "atr"],
                ["rsi", "macd", "macd_signal", "ema_20", "ema_50", "atr"],
                ["bollinger_mavg", "bollinger_hband", "bollinger_lband", "rsi", "cci", "atr"],
                ["atr", "bollinger_hband", "bollinger_lband", "adx", "momentum", "ema_50"],
                ["williams_r", "rsi", "macd", "macd_signal", "bollinger_mavg", "roc"],
                ["vwap", "ema_50", "ema_200", "adx", "sar", "senkou_span_a"]
            ]
            # Iterate over each set of features and train models
            for i, feature_set in enumerate(features):
                print(f"Training models with feature set {i}: {feature_set}")
                training_models.train_models(symbol, intervals, feature_set)

        except Exception as e:
            print(f"Error processing data for {symbol}: {e}")
            continue

        # Add a 30-second delay after each loop iteration
        time.sleep(10)
    print("Data processing complete.")



if __name__ == "__main__":
    run_machine_learning_and_historical_data()

# # ✅ Schedule every 60 minutes
# schedule.every(60).minutes.do(run_machine_learning_and_historical_data)

# while True:
#     schedule.run_pending()
#     time.sleep(1)