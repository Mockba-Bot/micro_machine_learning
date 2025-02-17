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
# # Access the environment variables
# PATH_OPERATIONS = os.getenv("PATH_OPERATIONS")

# # Add the path to the system path for custom modules
# sys.path.append(PATH_OPERATIONS)

# # Import custom operations module for database connection
# import operations
from database import operations

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

    table_exists = pd.read_sql(check_table_exists, operations.db_con).iloc[0, 0] > 0
    if table_exists:
        # Delete the last 15 rows to avoid duplicates
        delete_query = text(f"""
        DELETE FROM public."{tablename}"
        WHERE "timestamp" IN (
            SELECT "timestamp" 
            FROM public."{tablename}" 
            ORDER BY "timestamp" DESC 
            LIMIT 10
        );
        """)
        with operations.db_con.connect() as connection:
            connection.execute(delete_query)

        # Get the latest timestamp after deletion
        query = f'SELECT MAX("timestamp") as max_timestamp FROM public."{tablename}"'
        latest_timestamp = pd.read_sql(query, operations.db_con).iloc[0, 0]
    else:
        latest_timestamp = None

    # Determine the time range to fetch new data
    if latest_timestamp is not None:
        oldest_point = pd.to_datetime(latest_timestamp) + pd.Timedelta(minutes=1)  # Fetch from the next minute
    else:
        oldest_point = datetime.strptime("1 Jan 2017", "%d %b %Y")  # Default start date

    newest_point = datetime.utcnow()  # Current time

    # Fetch historical data from Binance
    klines = binance_client.get_historical_klines(
        symbol, kline_size,
        oldest_point.strftime("%d %b %Y %H:%M:%S"),
        newest_point.strftime("%d %b %Y %H:%M:%S")
    )

    data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close',
                                         'volume', 'close_time', 'quote_av', 'trades', 
                                         'tb_base_av', 'tb_quote_av', 'ignore'])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')

    # Remove rows with NaN values in the 'timestamp' column
    data.dropna(subset=['timestamp'], inplace=True)

    # Deduplicate the new data
    data = data.drop_duplicates(subset=['timestamp'])

    if save:
        # Deduplicate before saving
        # query_existing = f'SELECT timestamp FROM public."{tablename}"'
        # existing_timestamps = pd.read_sql(query_existing, operations.db_con)['timestamp']
        # data = data[~data['timestamp'].isin(existing_timestamps)]

        # Save new, non-duplicated data to the database
        data.to_sql(tablename, operations.db_con, if_exists='append', index=False)
        
    operations.remove_null_from_sql_table(tablename)
    return data


# Example usage
# get_all_binance("APTUSDT", "5m", "556159355", save=True)
