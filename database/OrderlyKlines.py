import os
import time
import requests
import urllib.parse
import pandas as pd
from base58 import b58decode
from base64 import urlsafe_b64encode
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from dotenv import load_dotenv
from datetime import datetime
from sqlalchemy import create_engine, TIMESTAMP, Float, String, text
import operations

# ✅ Load environment variables
dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env.micro.machine.learning'))
load_dotenv(dotenv_path=dotenv_path)

# ✅ Orderly API Config
BASE_URL = os.getenv("ORDERLY_BASE_URL")
ORDERLY_ACCOUNT_ID = os.getenv("ORDERLY_ACCOUNT_ID")
ORDERLY_SECRET = os.getenv("ORDERLY_SECRET")
ORDERLY_PUBLIC_KEY = os.getenv("ORDERLY_PUBLIC_KEY")

if not ORDERLY_SECRET or not ORDERLY_PUBLIC_KEY:
    raise ValueError("❌ ORDERLY_SECRET or ORDERLY_PUBLIC_KEY environment variables are not set!")

# ✅ Remove "ed25519:" prefix if present in private key
if ORDERLY_SECRET.startswith("ed25519:"):
    ORDERLY_SECRET = ORDERLY_SECRET.replace("ed25519:", "")

# ✅ Decode Base58 Private Key
private_key = Ed25519PrivateKey.from_private_bytes(b58decode(ORDERLY_SECRET))

# ✅ Fetch historical Orderly data with timestamp-based pagination
def fetch_historical_orderly(symbol, interval="1h", start_timestamp=None, save=True):
    """Fetch historical Orderly data using timestamp-based pagination."""
    
    # ✅ Ensure correct start date
    default_start_date = datetime(2023, 10, 26)  # Use their earliest available data
    if start_timestamp is None:
        start_timestamp = int(default_start_date.timestamp() * 1000)

    tablename = f"{symbol}_{interval}"

    # ✅ Check the latest available timestamp in the database
    with operations.db_con.connect() as conn:
        table_exists_query = text(f"""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = :tablename
            )
        """)
        table_exists = conn.execute(table_exists_query, {"tablename": tablename}).fetchone()[0]

        if table_exists:
            latest_timestamp_query = text(f"""
                SELECT MAX(end_timestamp) FROM public."{tablename}"
            """)
            result = conn.execute(latest_timestamp_query).fetchone()
            latest_timestamp = result[0] if result[0] else None

            if latest_timestamp:
                start_timestamp = int(latest_timestamp.timestamp() * 1000) + 1000  # Move forward

            # ✅ Delete last 20 rows to avoid redundant duplicates
            delete_query = text(f"""
            DELETE FROM public."{tablename}"
            WHERE "end_timestamp" IN (
                SELECT "end_timestamp" 
                FROM public."{tablename}" 
                ORDER BY "end_timestamp" DESC 
                LIMIT 20
            );
            """)
            conn.execute(delete_query)

    all_data = []
    last_timestamp = None
    no_change_count = 0  # Track if pagination is stuck

    while True:
        timestamp = str(int(time.time() * 1000))  # Generate new timestamp for request

        params = {
            "symbol": symbol,
            "type": interval,
            "start_timestamp": start_timestamp,  # Use pagination
            "limit": 1000,
        }

        path = "/v1/kline"
        query = f"?{urllib.parse.urlencode(params)}"
        message = f"{timestamp}GET{path}{query}"
        signature = urlsafe_b64encode(private_key.sign(message.encode())).decode()

        headers = {
            "orderly-timestamp": timestamp,
            "orderly-account-id": ORDERLY_ACCOUNT_ID,
            "orderly-key": ORDERLY_PUBLIC_KEY,
            "orderly-signature": signature,
        }

        url = f"{BASE_URL}{path}{query}"
        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            print(f"❌ Error fetching data: {response.json()}")
            break

        data = response.json().get("data", [])
        if not data:
            print("✅ No more historical data available.")
            break

        df = pd.DataFrame(data)

        # ✅ Handle cases where data is stored as a single column of dicts
        if df.shape[1] == 1 and isinstance(df.iloc[0, 0], dict):
            df = pd.json_normalize(df.iloc[:, 0])

        # ✅ Ensure all required columns exist
        required_columns = [
            "open", "close", "low", "high", "volume", "amount",
            "symbol", "type", "start_timestamp", "end_timestamp"
        ]
        
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"❌ Missing columns in response: {missing_cols}")

        # ✅ Convert timestamps to datetime
        df['start_timestamp'] = pd.to_datetime(df['start_timestamp'], unit='ms')
        df['end_timestamp'] = pd.to_datetime(df['end_timestamp'], unit='ms')

        all_data.append(df)

        # ✅ Ensure pagination continues correctly
        if 'end_timestamp' in df.columns:
            new_timestamp = df['end_timestamp'].iloc[-1]

            if last_timestamp == new_timestamp:
                no_change_count += 1

                if no_change_count >= 3:  # If stuck for 3 iterations, break the loop
                    break
            else:
                no_change_count = 0  # Reset counter if timestamp moves forward

            last_timestamp = new_timestamp
            start_timestamp = int(new_timestamp.timestamp() * 1000) + 1000  # Move **FORWARD** by 1 second
        else:
            print("⚠️ Missing 'end_timestamp'. Stopping data fetch.")
            break

    full_data = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

    # ✅ Save data to PostgreSQL if required
    if save and not full_data.empty:
        column_types = {
            "start_timestamp": TIMESTAMP,
            "end_timestamp": TIMESTAMP,
            "open": Float,
            "close": Float,
            "high": Float,
            "low": Float,
            "volume": Float,
            "amount": Float,
            "symbol": String,
            "type": String
        }

        # ✅ Remove duplicates before inserting new data
        full_data.drop_duplicates(subset=["start_timestamp", "symbol"], keep="last", inplace=True)

        # ✅ Append data to table, avoiding duplicates
        full_data.to_sql(tablename, operations.db_con, if_exists='append', index=False, dtype=column_types)
        print(f"✅ Data saved to database: {tablename}")

        # ✅ Remove NULL values from SQL table
        operations.remove_null_from_sql_table(tablename)

# ✅ Example Usage: Fetch Historical Data
fetch_historical_orderly("PERP_BTC_USDC", interval="1d")
