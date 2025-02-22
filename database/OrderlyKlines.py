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
import operations
from concurrent.futures import ThreadPoolExecutor, as_completed
import schedule

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
            symbols = [row["symbol"] for row in data["data"]["rows"] if "symbol" in row]
            return symbols
        else:
            print("⚠️ Unexpected API response format or missing 'data' key.")
            return []  # Return an empty list

    except Exception as e:
        print(f"❌ Error fetching Orderly symbols: {e}")
        return []  # Return an empty list
        
# ✅ Fetch historical Orderly data and insert into DB
def fetch_historical_orderly(symbol, interval="1h"):
    """Fetches the latest 1000 rows from Orderly, filters out existing data, and inserts only new records."""

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

    data = response.json().get("data", {})
    if not data or "rows" not in data:
        print("✅ No new data available.")
        return

    # ✅ Extract the "rows" array from the response
    rows = data.get("rows", [])
    df = pd.DataFrame(rows)

    # ✅ Ensure all required columns exist
    required_columns = ["start_timestamp", "open", "high", "low", "close", "volume"]
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"❌ Missing columns in response: {missing_cols}")

    # ✅ Convert timestamps to datetime and set index
    df['start_timestamp'] = pd.to_datetime(df['start_timestamp'], unit='ms')
    df.set_index('start_timestamp', inplace=True)  # ✅ Set start_timestamp as index

    # ✅ Filter the DataFrame to include only the required columns
    df = df[["open", "high", "low", "close", "volume"]]

    # ✅ Remove duplicates within the new data
    df = df[~df.index.duplicated(keep='first')]

    # ✅ Step 2: Check if the table exists
    with operations.db_con_historical.connect() as conn:
        table_exists_query = text(f"""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = :tablename
            )
        """)
        table_exists = conn.execute(table_exists_query, {"tablename": tablename}).fetchone()[0]
    conn.close()
    
    # ✅ Step 3: If table does not exist, create it
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

    # ✅ Step 4: Get existing timestamps if table exists
    existing_timestamps = []
    if table_exists:
        existing_timestamps_query = text(f'SELECT start_timestamp FROM "{tablename}"')
        existing_timestamps = pd.read_sql(existing_timestamps_query, operations.db_con_historical)['start_timestamp'].astype(str).tolist()  # Convert to str for comparison

    # ✅ Step 5: Filter out duplicates from df
    df.index = df.index.astype(str)  # Ensure same format
    df_filtered = df[~df.index.isin(existing_timestamps)]  # Keep only new rows

    # ✅ Step 6: Bulk insert only new rows
    if not df_filtered.empty:
        df_filtered.to_sql(tablename, operations.db_con_historical, if_exists="append", index=True, method="multi")
        print(f"✅ Inserted {len(df_filtered)} new rows into {tablename}.")
    else:
        print("✅ No new data to insert, all timestamps already exist.")

    # ✅ Step 7: Remove NULL values from SQL table (optional cleanup step)
    operations.remove_null_from_sql_table(tablename)

# ✅ Example Usage: Fetch Historical Data
orderly_symbols = fetch_orderly_symbols()
timeframes = ['1h', '4h', '1d', '5m']

# Process all intervals of 1h first, then 4h, and so on
def fetch_data_for_timeframe(timeframe):
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(fetch_historical_orderly, symbol, timeframe) for symbol in orderly_symbols]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"❌ Error fetching data: {e}")

def fetch_data():
    start_time = time.time()  # Track start time

    for timeframe in timeframes:
        print(f"🔄 Fetching historical data for all symbols - {timeframe}")
        fetch_data_for_timeframe(timeframe)
        time.sleep(1)  # Add a delay to avoid rate limits

    end_time = time.time()  # Track end time
    elapsed_time = end_time - start_time
    print(f"✅ Data fetching complete. Total time taken: {elapsed_time:.2f} seconds.")   

# Run the function immediately when the script starts
print("🚀 Running initial fetch...")
fetch_data()

# Schedule the function to run every 10 minutes
schedule.every(10).minutes.do(fetch_data)

# Keep the script running
while True:
    schedule.run_pending()
    time.sleep(1)  # Sleep for a short time to avoid high CPU usage

