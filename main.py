import sys
import os
from dotenv import load_dotenv
import redis
import requests
from datetime import timedelta
# Add the directory containing your modules to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'machine_learning')))
from machine_learning import bucket, elliot_waves_analysis, scalping_models, signal_models, technical_analysis, training_models
from database import getHistorical
import time  # Import time module


# Load environment variables from the .env file
load_dotenv(dotenv_path=".env.micro.machine.learning")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost")
# Initialize Redis connection
redis_client = redis.from_url(REDIS_URL, decode_responses=True)  


def get_all_binance_symbols():
    redis_key = "binance_usdt_symbols"

    # Check Redis for cached symbols
    cached_symbols = redis_client.get(redis_key)
    if cached_symbols:
        return cached_symbols.split(',')

    # Fetch data from Binance API
    url = "https://api.binance.com/api/v3/exchangeInfo"
    try:
        response = requests.get(url)
        data = response.json()

        # Filter symbols that are active for spot trading and end with 'USDT'
        symbols = [
            symbol['symbol'] for symbol in data['symbols'] 
            if symbol['status'] == 'TRADING' and symbol['symbol'].endswith('USDT')
        ]

        # Store the filtered symbols in Redis with a 1-month expiration
        redis_client.set(redis_key, ",".join(symbols))  # Store symbols as a comma-separated string
        redis_client.expire(redis_key, timedelta(days=30))  # Set expiration to 30 days

        # Return the filtered symbols
        return symbols
        # return [symbol for symbol in symbols if symbol == 'APTUSDT']
    except Exception as e:
        print(f"Error fetching or storing symbols: {e}")
        return []


if __name__ == "__main__":
    symbols = get_all_binance_symbols()
    for symbol in symbols:
        if symbol == 'APTUSDT':
            print(f"Fetching data for {symbol}...")
            intervals = ['1h', '4h', '1d']
            getHistorical.get_all_binance(symbol, '1h', '000000', True)
            time.sleep(3)
            getHistorical.get_all_binance(symbol, '4h', '000000', True)
            time.sleep(3)
            getHistorical.get_all_binance(symbol, '1d', '000000', True)
            time.sleep(3)
            try:
                # Training for Elliot Waves
                print("---TRAINING ELLIOT WAVES---")
                elliot_waves_analysis.train_models(symbol, intervals)
                time.sleep(5)

                print("---TRAINING SCALPING MODELS---")
                # Training for scalping models
                stop_loss_percentage = 0.5 # 50% stop loss
                profit_target_from = 0.1 # 1% profit target
                profit_target_to = 0.3 # 3% profit target
                partial_exit_threshold_from = 25.0 # 25% partial exit threshold
                partial_exit_threshold_to = 30.0 # 30% partial exit threshold
                exit_remaining_percentage_from = 15.0 # 15% exit remaining percentage
                exit_remaining_percentage_to = 20.0 # 20% exit remaining percentage
                partial_exit_amount = 0.15 # 15% partial exit amount
                scalping_models.train_scalping_models('000000', symbol, '5m', stop_loss_percentage, profit_target_from, profit_target_to, partial_exit_threshold_from, partial_exit_threshold_to, exit_remaining_percentage_from, exit_remaining_percentage_to, partial_exit_amount)
                time.sleep(5)

                print("---TRAINING SIGNAL MODELS---")
                # Training for signal models
                signal_models.train_models(symbol, intervals)
                time.sleep(5)

                print("---TRAINING TECHNICAL ANALYSIS---")
                # Training for technical analysis
                technical_analysis.train_models(symbol, intervals)
                time.sleep(5)

                print("---TRAINING MACHINE LEARNING MODELS---")
                # Training for machine learning models
                training_models.train_models(symbol, intervals)

            except Exception as e:
                print(f"Error processing data for {symbol}: {e}")
                continue

            # Add a 30-second delay after each loop iteration
            time.sleep(30)
    print("Data processing complete.")