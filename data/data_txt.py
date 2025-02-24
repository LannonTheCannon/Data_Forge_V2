import sqlite3
import pandas as pd

DB_FILE = "../fraud_data.db"
TABLE_NAME = "fraud_data"
OUTPUT_TXT = "fraud_data.txt"
LIMIT = 1000

def convert_db_to_txt():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME} LIMIT {LIMIT};", conn)
    conn.close()

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            # Convert each row to a single line of text
            # Example: "id=123, amount=450, date=2023-08-01, fraud=True"
            line_str = ", ".join(f"{col}={row[col]}" for col in df.columns)
            f.write(line_str + "\n")

    print(f"✅ Created {OUTPUT_TXT}. Each line is one row from {TABLE_NAME}.")

if __name__ == "__main__":
    convert_db_to_txt()
