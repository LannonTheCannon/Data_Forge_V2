import sqlite3
import pandas as pd

DB_FILE = "../fraud_data.db"
TABLE_NAME = "fraud_data"
LIMIT = 1000

def export_db_to_csv():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME} LIMIT {LIMIT};", conn)
    conn.close()

    # 1) Convert each row into a single text string if you want.
    #    For example, "id=123, amount=400.0, fraud=True, date=2023-08-01"
    #    or you can just keep normal columns—OpenAI might try to embed each row as is.
    df_for_embedding = pd.DataFrame()
    df_for_embedding["text"] = df.apply(
        lambda row: ", ".join(f"{col}={row[col]}" for col in df.columns),
        axis=1
    )

    # 2) Save to a CSV file (one row = one "document")
    df_for_embedding.to_csv("fraud_data.csv", index=False)
    print("✅ Created fraud_data.csv with a 'text' column.")


export_db_to_csv()
