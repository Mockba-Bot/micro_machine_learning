import sys
import os
from sqlalchemy import text
import pandas as pd
# Add the directory containing your modules to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from database import operations

# Fetch historical data from the database
def get_historical_data(pair, timeframe, values):
    table = f'"{pair}_{timeframe}"'
    f, t = values.split('|')
    
    query = text(f"""
        SELECT start_timestamp, low, high, volume, close 
        FROM public.{table} 
        WHERE timestamp >= :start_time AND timestamp <= :end_time 
        ORDER BY 1
    """)
    
    df = pd.read_sql(query, con=operations.db_con_historical, params={"start_time": f, "end_time": t})
    
    # Convert columns to numeric types
    df['close'] = pd.to_numeric(df['close'])
    df['high'] = pd.to_numeric(df['high'])
    df['low'] = pd.to_numeric(df['low'])
    df['volume'] = pd.to_numeric(df['volume'])
    
    return df

# Fetch historical data from the database
def get_historical_data_limit(pair, timeframe, limit=200):
    table = f'"{pair}_{timeframe}"'
    
    query = text(f"""
        SELECT start_timestamp, low, high, volume, close 
        FROM public.{table} 
        ORDER BY 1
        LIMIT {limit}
    """)
    
    df = pd.read_sql(query, con=operations.db_con_historical)
    
    # Convert columns to numeric types
    df['close'] = pd.to_numeric(df['close'])
    df['high'] = pd.to_numeric(df['high'])
    df['low'] = pd.to_numeric(df['low'])
    df['volume'] = pd.to_numeric(df['volume'])
    
    return df    