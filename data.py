import pickle
from pathlib import Path

import pandas as pd
import streamlit as st


# def load_file(path: str) -> pd.DataFrame:
#     with open(path, "rb") as f:
#         dataset = pickle.load(f)
#         return dataset
#
#
# @st.cache_data
# def load_data(folder: str) -> pd.DataFrame:
#     all_datasets = [load_file(file) for file in Path(folder).iterdir()]
#     df = pd.concat(all_datasets)
#     return df

import sqlite3
import pandas as pd
import streamlit as st

DB_FILE = "fraud_data.db"


@st.cache_data
def load_data() -> pd.DataFrame:
    """Load data from SQLite instead of pickle files."""
    conn = sqlite3.connect(DB_FILE)

    # Load only what is needed (OPTIONAL: Add WHERE clauses for filters)
    query = "SELECT * FROM fraud_data LIMIT 100000;"  # Change LIMIT based on performance
    df = pd.read_sql(query, conn)

    conn.close()
    return df
