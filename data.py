import os
import requests
import sqlite3
import pandas as pd
import gdown

DB_FILE = "fraud_data.db"
file_id = "1RQu28-etwF4BcO62-Lr4gnZPkul6pCnJ"  # Replace with actual File ID
output = "fraud_data.db"

# DRIVE_URL = f"https://drive.google.com/uc?export=download&id={DRIVE_FILE_ID}"

def download_database():
    """Downloads the database file from Google Drive."""
    print("⬇️ Downloading database file from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)

    # Check if the file exists and has a reasonable size
    if os.path.exists(output) and os.path.getsize(output) > 0:
        print(f"✅ Database successfully downloaded. File size: {os.path.getsize(output)} bytes")
    else:
        print("❌ Failed to download a valid database file.")



download_database()


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