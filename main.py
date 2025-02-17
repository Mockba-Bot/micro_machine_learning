import sys
import os
from dotenv import load_dotenv
import redis
import requests
from datetime import timedelta

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
        # return symbols
        return [symbol for symbol in symbols if symbol == 'BTCUSDT']
    except Exception as e:
        print(f"Error fetching or storing symbols: {e}")
        return []


if __name__ == "__main__":
    get_all_binance_symbols()