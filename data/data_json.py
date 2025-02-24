import json
import sqlite3
import pandas as pd

DB_FILE = "../fraud_data.db"
conn = sqlite3.connect(DB_FILE)
df = pd.read_sql("SELECT * FROM fraud_data LIMIT 1000;", conn)
conn.close()

with open("fraud_data.jsonl", "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        # Convert to a "document" field or something similar
        # For example: { "text": "UserID=123; Amount=400; Fraud=True" }
        doc = {"text": ", ".join(f"{k}={v}" for k, v in row_dict.items())}
        f.write(json.dumps(doc) + "\n")