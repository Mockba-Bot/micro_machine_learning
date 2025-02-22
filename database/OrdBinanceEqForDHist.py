import os
import requests
import urllib.parse
import pandas as pd
from datetime import timedelta
from sqlalchemy import create_engine, text, String
from dotenv import load_dotenv
from getHistorical import get_all_binance
import time

# ✅ Load environment variables
dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env.micro.machine.learning'))
load_dotenv(dotenv_path=dotenv_path)

# ✅ Database Configuration
DB_URL = os.getenv("DATABASE_URL")  # PostgreSQL connection URL
engine = create_engine(DB_URL)

# ✅ Orderly API Config
BASE_URL = "https://api-evm.orderly.org"

# ✅ Function to fetch Binance symbols
def get_all_binance_symbols():
    # ✅ Fetch data from Binance API
    url = "https://api.binance.com/api/v3/exchangeInfo"
    try:
        response = requests.get(url)
        data = response.json()

        # ✅ Filter active symbols that end with 'USDT'
        symbols = [
            symbol['symbol'][:-4] for symbol in data['symbols']
            if symbol['status'] == 'TRADING' and symbol['symbol'].endswith('USDT')
        ]

        return symbols
    except Exception as e:
        print(f"❌ Error fetching Binance symbols: {e}")
        return []

# ✅ Fetch Orderly Trading Pairs
def fetch_orderly_symbols():
    url = f"{BASE_URL}/v1/public/info"
    try:
        # Make the API request
        response = requests.get(url)
        
        # Parse JSON response
        data = response.json()

        # Check if the response is successful and contains the "data" key
        if data.get("success") and "data" in data:
            # Extract the list of symbols from the "rows" array
            symbols = [row["symbol"][:-5].replace("PERP_","") for row in data["data"]["rows"] if "symbol" in row]
            return symbols
        else:
            print("⚠️ Unexpected API response format or missing 'data' key.")
            return []  # Return an empty list

    except Exception as e:
        print(f"❌ Error fetching Orderly symbols: {e}")
        return []  # Return an empty list

def get_common_symbols():
    try:
        # Fetch symbols from Orderly
        orderly_symbols = fetch_orderly_symbols()
        # print(f"Orderly Symbols: {orderly_symbols}")

        # Fetch symbols from Binance
        binance_symbols = get_all_binance_symbols()
        # print(f"Binance Symbols: {binance_symbols}")

        # Find common symbols using set intersection
        common_symbols = list(set(orderly_symbols) & set(binance_symbols))
        return common_symbols

    except Exception as e:
        print(f"❌ Error comparing symbols: {e}")
        return []       

# def drop_perp_tables():
#     with engine.connect() as conn:
#         # Fetch all table names that start with 'PERP'
#         query = text("""
#             SELECT table_name
#             FROM information_schema.tables
#             WHERE table_schema = 'public'
#             AND table_name LIKE 'PERP%';
#         """)
#         result = conn.execute(query)
#         tables_to_drop = [row[0] for row in result]  # Access the first element of the tuple

#         # Drop each table
#         for table in tables_to_drop:
#             drop_query = text(f'DROP TABLE IF EXISTS public."{table}" CASCADE;')
#             conn.execute(drop_query)
#             print(f"🗑️ Dropped table: {table}")

#         conn.commit()

# # Run the function to drop tables
# drop_perp_tables()

# ✅ Run the function
common_symbols = get_common_symbols()
# Concatenate "USDT" to each common symbol
common_symbols_with_usdt = [symbol + "USDT" for symbol in common_symbols]

# Sort the symbols alphabetically
common_symbols_with_usdt = sorted(common_symbols_with_usdt)

timeframes = ['1h', '4h', '1d', '5m']

# print(f"Common Symbols with USDT: {common_symbols_with_usdt}")

# Run get_all_binance for all common symbols and intervals
for symbol in common_symbols_with_usdt:
    for timeframe in timeframes:
        print(f"Fetching data for {symbol} with timeframe {timeframe}...")
        get_all_binance(symbol, timeframe, '000000', True)
        time.sleep(3)  # Add a delay to avoid hitting rate limits