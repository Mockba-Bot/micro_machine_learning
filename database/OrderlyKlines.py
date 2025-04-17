import os
import time
import requests
import urllib.parse
import pandas as pd
from base58 import b58decode
from base64 import urlsafe_b64encode
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from dotenv import load_dotenv
from sqlalchemy import TIMESTAMP, Float, text
# from database import operations
import operations
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from datetime import datetime, timedelta, timezone

# ✅ Load environment variables
dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env.micro.machine.learning'))
load_dotenv(dotenv_path=dotenv_path)


# ✅ Orderly API Config
BASE_URL = os.getenv("ORDERLY_BASE_URL")
ORDERLY_ACCOUNT_ID = os.getenv("ORDERLY_ACCOUNT_ID")
ORDERLY_SECRET = os.getenv("ORDERLY_SECRET")
ORDERLY_PUBLIC_KEY = os.getenv("ORDERLY_PUBLIC_KEY")
MAX_WORKERS = int(os.getenv("MAX_WORKERS", 10))

if not ORDERLY_SECRET or not ORDERLY_PUBLIC_KEY:
    raise ValueError("❌ ORDERLY_SECRET or ORDERLY_PUBLIC_KEY environment variables are not set!")

# ✅ Remove "ed25519:" prefix if present in private key
if ORDERLY_SECRET.startswith("ed25519:"):
    ORDERLY_SECRET = ORDERLY_SECRET.replace("ed25519:", "")

# ✅ Decode Base58 Private Key
private_key = Ed25519PrivateKey.from_private_bytes(b58decode(ORDERLY_SECRET))

# ✅ Define available timeframes
timeframes = ['1h', '4h', '1d', '5m']

# ✅ Rate limiter (Ensures max 8 API requests per second globally)
class RateLimiter:
    def __init__(self, max_calls, period):
        self.max_calls = max_calls
        self.period = period
        self.calls = []
        self.lock = threading.Lock()

    def __call__(self):
        with self.lock:
            now = time.time()
            self.calls = [call for call in self.calls if call > now - self.period]
            if len(self.calls) >= self.max_calls:
                sleep_time = self.period - (now - self.calls[0])
                print(f"⏳ Rate limit reached! Sleeping for {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            self.calls.append(time.time())

# ✅ Initialize Global Rate Limiter
rate_limiter = RateLimiter(max_calls=10, period=1)

# ✅ Fetch Orderly Trading Pairs
def fetch_orderly_symbols():
    url = f"{BASE_URL}/v1/public/info"
    try:
        response = requests.get(url)
        data = response.json()
        if data.get("success") and "data" in data:
            one_month_ago = datetime.now(timezone.utc) - timedelta(days=30)

            symbols = []
            for row in data["data"]["rows"]:
                created_time = row.get("created_time")
                if created_time:
                    created_datetime = datetime.fromtimestamp(created_time / 1000, tz=timezone.utc)
                    if created_datetime <= one_month_ago and "symbol" in row:
                        symbols.append(row["symbol"])
            return symbols
        else:
            print("⚠️ Unexpected API response format or missing 'data' key.")
            return []
    except Exception as e:
        print(f"❌ Error fetching Orderly symbols: {e}")
        return []
        
# ✅ Fetch historical Orderly data with global rate limiting
def fetch_historical_orderly(symbol, interval):
    rate_limiter()  # ✅ Apply global rate limit

    timestamp = str(int(time.time() * 1000))
    params = {"symbol": symbol, "type": interval, "limit": 1000}
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
        print(f"❌ Error fetching data for {symbol} {interval}: {response.json()}")
        return None

    data = response.json().get("data", {})
    if not data or "rows" not in data:
        return None

    df = pd.DataFrame(data["rows"])
    required_columns = ["start_timestamp", "open", "high", "low", "close", "volume"]
    if set(required_columns).issubset(df.columns):
        df['start_timestamp'] = pd.to_datetime(df['start_timestamp'], unit='ms')
        df.set_index('start_timestamp', inplace=True)
        df = df[["open", "high", "low", "close", "volume"]]
        df = df[~df.index.duplicated(keep='first')]  # Remove duplicates
        return df
    return None

# ✅ Fetch all timeframes for a symbol (Now Sequential)
def fetch_symbol_data(symbol, interval):
    """Fetches all timeframes for a symbol sequentially to avoid rate limit errors."""
    symbol_data = {}

    # print(f"📥 Fetching {symbol} {interval} data...")
    df = fetch_historical_orderly(symbol, interval)
    if df is not None:
        symbol_data[interval] = df

    return symbol_data

# ✅ Process data storage in parallel
def store_data(symbol, interval, df):
    tablename = f"{symbol}_{interval}"
    
    # Step 1: Check if table exists
    with operations.db_con_historical.connect() as conn:
        table_exists_query = text(f"SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = :tablename)")
        table_exists = conn.execute(table_exists_query, {"tablename": tablename}).fetchone()[0]
        
        # Step 2: Create table if not exists
        if not table_exists:
            print(f"🛠️ Table {tablename} does not exist. Creating it now...")
            column_types = {
                "start_timestamp": TIMESTAMP,
                "open": Float,
                "high": Float,
                "low": Float,
                "close": Float,
                "volume": Float
            }
            df.head(0).to_sql(tablename, operations.db_con_historical, if_exists='fail', index=True, dtype=column_types)
            
            # Create index on start_timestamp for new tables
            conn.execute(text(f'CREATE INDEX idx_{tablename}_timestamp ON "{tablename}" (start_timestamp)'))
            print(f"✅ Created index on start_timestamp for {tablename}")

    # Step 3-5: Existing data insertion logic remains the same
    existing_timestamps = []
    if table_exists:
        existing_timestamps_query = text(f'SELECT start_timestamp FROM "{tablename}"')
        existing_timestamps = pd.read_sql(existing_timestamps_query, operations.db_con_historical)['start_timestamp'].astype(str).tolist()

    df.index = df.index.astype(str)
    df_filtered = df[~df.index.isin(existing_timestamps)]

    if not df_filtered.empty:
        df_filtered.to_sql(tablename, operations.db_con_historical, if_exists="append", index=True, method="multi")

    operations.remove_null_from_sql_table(tablename)

def add_indexes_to_existing_tables():
    with operations.db_con_historical.connect() as conn:
        # Get all tables following your naming convention
        tables = pd.read_sql("SELECT table_name FROM information_schema.tables WHERE table_name LIKE 'PERP_%_%'", conn)
        
        for table in tables['table_name']:
            try:
                conn.execute(text(f'CREATE INDEX idx_{table}_timestamp ON "{table}" (start_timestamp)'))
                print(f"✅ Added index to {table}")
            except Exception as e:
                print(f"⚠️ Couldn't add index to {table}: {str(e)}")

# ✅ Fetch and store historical data efficiently
def fetch_and_store(interval, orderly_symbols):
    # orderly_symbols = fetch_orderly_symbols()
    all_data = {}
    batch_size = 10  # ✅ Process 10 symbols in parallel

    # print(f"📥 Fetching symbols data...")

    for i in range(0, len(orderly_symbols), batch_size):
        batch = orderly_symbols[i:i + batch_size]
        # print(f"🔁 Processing batch: {batch}")

        # ✅ Fetch data in parallel (Each batch runs in parallel)
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = {executor.submit(fetch_symbol_data, symbol, interval): symbol for symbol in batch}

            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    all_data[symbol] = future.result()
                except Exception as e:
                    print(f"❌ Error fetching data for {symbol}: {e}")

        time.sleep(1)  # ✅ Prevents API overload

    # print(f"📦 Storing fetched data...")
    futures = []
    with ThreadPoolExecutor(max_workers=min(len(all_data), MAX_WORKERS)) as executor:
        for symbol, data in all_data.items():
            if interval in data and isinstance(data[interval], pd.DataFrame):
                df = data[interval]
                futures.append(executor.submit(store_data, symbol, interval, df))
            else:
                print(f"⚠️ No valid data found for {symbol} at interval {interval}. Skipping.")

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"❌ Error inserting data: {e}")

    

# ✅ Loop through each timeframe and process it separately
def run_all_timeframes(orderly_symbols):
    print(f"📥 Fetching symbols data...")
    start_time = time.time()
    for timeframe in timeframes:
        fetch_and_store(timeframe, orderly_symbols)
    end_time = time.time()
    print(f"✅ Data fetched and stored in {end_time - start_time:.2f} seconds.")


# symbol = 'PERP_APT_USDC'
# interval = '1h'
# df = fetch_historical_orderly(symbol, interval)
# store_data(symbol, interval, df)
# # add_indexes_to_existing_tables