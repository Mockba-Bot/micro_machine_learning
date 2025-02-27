# IMPORTS
import sys
import os
import pandas as pd
import math
import os.path
import time
from binance.client import Client
from sqlalchemy.sql import text
from datetime import datetime
from dateutil import parser
from dotenv import load_dotenv
# import operations
from database import operations
# ✅ Load environment variables
dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env.micro.machine.learning'))
load_dotenv(dotenv_path=dotenv_path)

### API
def get_binance_client(api_telegram):
    binance_api_key = '000000'
    binance_api_secret = '000000'
    return Client(api_key=binance_api_key, api_secret=binance_api_secret)

### CONSTANTS
binsizes = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "2h": 120, "4h": 240, "1d": 1440}
batch_size = 750
date = datetime(2023, 1, 1).strftime("%d %b %Y")

### FUNCTIONS
def minutes_of_new_data(symbol, kline_size, data, source, binance_client):
    if len(data) > 0:
        old = parser.parse(data.index[-1].strftime("%Y-%m-%d %H:%M:%S"))
    elif source == "binance":
        old = datetime.strptime(date, '%d %b %Y')
    if source == "binance":
        new = pd.to_datetime(binance_client.get_klines(
            symbol=symbol, interval=kline_size)[-1][0], unit='ms')
    return old, new


def get_all_binance(symbol, kline_size, token, save=False):
    binance_client = get_binance_client(token)
    if binance_client is None:
        return None

    tablename = f"{symbol}_{kline_size}"
    
    check_table_exists = f"SELECT count(*) FROM information_schema.tables WHERE table_name = '{tablename}';"

    table_exists = pd.read_sql(check_table_exists, operations.db_con_historical).iloc[0, 0] > 0
    if table_exists:
        # Delete the last 15 rows to avoid duplicates
        delete_query = text(f"""
        DELETE FROM public."{tablename}"
        WHERE "start_timestamp" IN (
            SELECT "start_timestamp" 
            FROM public."{tablename}" 
            ORDER BY "start_timestamp" DESC 
            LIMIT 10
        );
        """)
        with operations.db_con_historical.connect() as connection:
            connection.execute(delete_query)

        # Get the latest timestamp after deletion
        query = f'SELECT MAX("start_timestamp") as max_timestamp FROM public."{tablename}"'
        latest_timestamp = pd.read_sql(query, operations.db_con_historical).iloc[0, 0]
    else:
        latest_timestamp = None

    # Determine the time range to fetch new data
    if latest_timestamp is not None:
        oldest_point = pd.to_datetime(latest_timestamp) + pd.Timedelta(minutes=1)  # Fetch from the next minute
    else:
        oldest_point = datetime.strptime("1 Jan 2023", "%d %b %Y")  # Default start date

    newest_point = datetime.utcnow()  # Current time

    # Fetch historical data from Binance
    klines = binance_client.get_historical_klines(
        symbol, kline_size,
        oldest_point.strftime("%d %b %Y %H:%M:%S"),
        newest_point.strftime("%d %b %Y %H:%M:%S")
    )

    data = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 
        'close_time', 'quote_asset_volume', 'number_of_trades', 
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    
    # Remove rows with NaN values in the 'timestamp' column
    data.dropna(subset=['timestamp'], inplace=True)

    # Deduplicate the new data
    data = data.drop_duplicates(subset=['timestamp'])

    if save:
        # Select only the required columns
        data = data[['open', 'high', 'low', 'close', 'timestamp', 'volume']]
        
        # Rename 'timestamp' to 'start_timestamp'
        data.rename(columns={'timestamp': 'start_timestamp'}, inplace=True)

        # Set 'start_timestamp' as the index
        data.set_index('start_timestamp', inplace=True)

        # Save new, non-duplicated data to the database
        symbol = "PERP_"+symbol.replace("USDT", "_USDC")
        tablename = f"{symbol}_{kline_size}"
        data.to_sql(tablename, operations.db_con_historical, if_exists='append', index=True)
        
    operations.remove_null_from_sql_table(tablename)
    return data


# Example usage
# get_all_binance("APTUSDT", "5m", "556159355", save=True)
