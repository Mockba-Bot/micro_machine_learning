import os
import time
import requests
import urllib.parse
import pandas as pd
from base58 import b58decode
from base64 import urlsafe_b64encode
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from dotenv import load_dotenv
from sqlalchemy import TIMESTAMP, Float, String, text
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

# ✅ Fetch historical Orderly data and insert into DB
def fetch_historical_orderly(symbol, interval="1h"):
    """Fetches the latest 1000 rows from Orderly, deletes last 50 rows in DB, and inserts new records."""

    tablename = f"{symbol}_{interval}"

    # ✅ Step 1: Fetch latest 1000 rows from Orderly
    timestamp = str(int(time.time() * 1000))  # Generate new timestamp for request
    params = {
        "symbol": symbol,
        "type": interval,
        "limit": 1000,  # Orderly always returns the latest 1000 rows
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
        return

    data = response.json().get("data", [])
    if not data:
        print("✅ No new data available.")
        return

    df = pd.DataFrame(data)

    # ✅ Handle cases where data is stored as a single column of dicts
    if df.shape[1] == 1 and isinstance(df.iloc[0, 0], dict):
        df = pd.json_normalize(df.iloc[:, 0])

    # ✅ Ensure all required columns exist
    required_columns = [
        "open", "close", "low", "high", "volume", "amount",
        "symbol", "type", "start_timestamp"
    ]
    
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"❌ Missing columns in response: {missing_cols}")

    # ✅ Convert timestamps to datetime and set index
    df['start_timestamp'] = pd.to_datetime(df['start_timestamp'], unit='ms')
    df.set_index("start_timestamp", inplace=True)

    # ✅ Step 2: Delete last 50 rows in DB to avoid duplicates
    with operations.db_con.connect() as conn:
        table_exists_query = text(f"""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = :tablename
            )
        """)
        table_exists = conn.execute(table_exists_query, {"tablename": tablename}).fetchone()[0]

        if table_exists:
            print(f"🛠️ Deleting last 50 rows from {tablename} to prevent duplication...")
            delete_query = text(f"""
                DELETE FROM public."{tablename}"
                WHERE start_timestamp IN (
                    SELECT start_timestamp FROM public."{tablename}"
                    ORDER BY start_timestamp DESC 
                    LIMIT 50
                );
            """)
            conn.execute(delete_query)

    # ✅ Step 3: Insert new data into DB
    column_types = {
        "open": Float,
        "close": Float,
        "high": Float,
        "low": Float,
        "volume": Float,
        "amount": Float,
        "symbol": String,
        "type": String
    }

    # ✅ If table doesn't exist, create it
    if not table_exists:
        print(f"🛠️ Table {tablename} does not exist. Creating it now...")
        df.head(0).to_sql(tablename, operations.db_con, if_exists='fail', index=True, dtype=column_types)

    # ✅ Insert new records
    df.to_sql(tablename, operations.db_con, if_exists='append', index=True, dtype=column_types)
    print(f"✅ Data saved to database: {tablename}")

    # ✅ Step 4: Remove NULL values from SQL table
    operations.remove_null_from_sql_table(tablename)

# ✅ Example Usage: Fetch Historical Data
fetch_historical_orderly("PERP_BTC_USDC", interval="5m")
