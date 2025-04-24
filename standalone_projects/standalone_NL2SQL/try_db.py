import pandas as pd
from sqlalchemy import create_engine

DB_PATH = "../../data/northwind.db"
engine = create_engine(f"sqlite:///{DB_PATH}")

with engine.connect() as conn:
    df = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
    print(df)

df = pd.read_sql("SELECT * FROM Orders LIMIT 5;", conn)
print(df.head())