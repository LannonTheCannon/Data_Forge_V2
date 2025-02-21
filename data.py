import os
import requests
import sqlite3
import pandas as pd

DB_FILE = "fraud_data.db"
DRIVE_FILE_ID = "1RQu28-etwF4BcO62-Lr4gnZPkul6pCnJ"  # Replace with actual File ID
DRIVE_URL = f"https://drive.google.com/uc?export=download&id={DRIVE_FILE_ID}"

def download_db():
    """Download the SQLite database if not present locally."""
    if not os.path.exists(DB_FILE):
        print("Downloading database from Google Drive...")
        response = requests.get(DRIVE_URL, timeout=60)
        with open(DB_FILE, "wb") as f:
            f.write(response.content)
        print("Database downloaded successfully.")


download_db()


def load_data():
    """Load the SQLite database."""
    print(f"Checking if database file exists: {os.path.exists(DB_FILE)}")

    if not os.path.exists(DB_FILE):
        raise FileNotFoundError(f"Database file {DB_FILE} not found!")

    try:
        conn = sqlite3.connect(DB_FILE)
        print("✅ Database successfully opened")
        df = pd.read_sql("SELECT * FROM fraud_data LIMIT 10000;", conn)  # Adjust query as needed
        conn.close()
        return df
    except sqlite3.DatabaseError as e:
        print("❌ Database error: ", e)
        raise e
