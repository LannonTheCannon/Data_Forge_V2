import sqlite3
import pickle
import pandas as pd
from pathlib import Path

DB_FILE = "fraud_data.db"


def load_file(path: str) -> pd.DataFrame:
    """Load data from a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def save_to_sqlite(folder: str, db_file: str = DB_FILE):
    """Convert all .pkl files into an SQLite database."""
    conn = sqlite3.connect(db_file)

    # Load and store each file into SQLite
    for i, file in enumerate(Path(folder).iterdir()):
        df = load_file(file)

        # Save DataFrame to SQLite
        df.to_sql("fraud_data", conn, if_exists="append", index=False)

        print(f"✔ Saved {file.name} to SQLite ({i + 1}/{len(list(Path(folder).iterdir()))})")

    conn.close()
    print(f"✔ All data stored in {db_file}")


# Run this once to create the SQLite3 database
save_to_sqlite("./data")
