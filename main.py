import sys
import os
from dotenv import load_dotenv
import requests
import time  # Import time module
import schedule

# Add the directory containing your modules to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'machine_learning')))
from machine_learning import bucket, elliot_waves_analysis, signal_models, technical_analysis, training_models
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
        intervals = ['1h', '4h', '1d', '5m']
        strategies = ["Trend-Following", "Volatility Breakout", "Momentum Reversal", "Momentum + Volatility", "Hybrid", "Advanced", "Router"]
        try:
            # Training for Elliot Waves
            print("---TRAINING ELLIOT WAVES---")
            elliot_waves_analysis.train_models(symbol, intervals)
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
            training_models.train_models(symbol, intervals, strategies)
            time.sleep(2)
            
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