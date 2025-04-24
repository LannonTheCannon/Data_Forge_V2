import pandas as pd
from sqlalchemy import create_engine

DB_PATH = "../../data/northwind.db"
engine = create_engine(f"sqlite:///{DB_PATH}")

with engine.connect() as conn:
    # ✅ Get list of all tables
    df_tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
    print("Tables in DB:\n", df_tables)

    # ✅ Try querying Orders table
    df_orders = pd.read_sql("SELECT * FROM Orders LIMIT 5;", conn)
    print("\nSample Orders:\n", df_orders.head())