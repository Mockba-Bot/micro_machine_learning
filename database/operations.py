import os
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine
import psycopg2

# Load environment variables from the .env file
dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env.micro.machine.learning'))
load_dotenv(dotenv_path=dotenv_path)

# access the environment variables
host = os.getenv("HOST")
database = os.getenv("DATABASE")
user = os.getenv("USR")
password = os.getenv("PASSWD")

 
db_con = create_engine(f"postgresql://{user}:{password}@{host}:5432/{database}")  


# Drop duplicates from SQL table Historical
def remove_null_from_sql_table(table_name):
    try:
        conn = psycopg2.connect(host=host, database=database, user=user, password=password)
        cursor = conn.cursor()
        # Delete rows with null timestamp from the specified table
        sql = f"""
        DELETE FROM public."{table_name}"
        WHERE start_timestamp is null or end_timestamp is null;
        """
        cursor.execute(sql)
        conn.commit()
    except psycopg2.Error as e:
        print(f"Error: {e}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

