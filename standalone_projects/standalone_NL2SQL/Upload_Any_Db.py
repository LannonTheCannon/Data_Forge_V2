import tempfile
import streamlit as st
import os
import sqlalchemy as sql
from sqlalchemy.pool import NullPool

st.sidebar.header('User your own database')

# 1) File uploader for SQLite files
upload = st.sidebar.file_uploader(
    "Upload a SQLite DB file",
    type=['sqlite', 'db', 'sqlite3']
)

# Let them paste any SQLAlchemy URL
conn_url = st.sidebar.text_input(
    "Or enter a SQLAlchemy URL",
    placeholder='e.g. postgresql://user:password@localhost:5432/dbname'
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 1) Point at your local files
NW_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "data", "northwind.db"))
CH_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "data", "Chinook_Sqlite.sqlite"))

# 2) Map a friendly name → filesystem path
DB_FILES = {
    "Northwind Database": NW_PATH,
    "Chinook (SQLite)": CH_PATH,
}

# 3) Build the SQLAlchemy URLs from those paths
DB_OPTIONS = {
    name: f"sqlite:///{path}"
    for name, path in DB_FILES.items()
}

# Determine which one to use
if upload is not None:
    # Save the uploaded bytes into a temp file
    tmp = tempfile.NamedTemporaryFile(suffix="_"+upload.name, delete=False)
    tmp.write(upload.read())
    tmp.flush()
    raw_path = tmp.name
    engine_url = f"sqlite:///{raw_path}"

elif conn_url:
    # user-provided URL
    engine_url = conn_url
    # for display/existence check:
    raw_path = conn_url.split("///")[-1] if conn_url.startswith("sqlite") else conn_url

else:
    # fallback to your built-in choices
    choice = st.sidebar.selectbox("Select sample DB", list(DB_OPTIONS.keys()))
    engine_url = DB_OPTIONS[choice]
    raw_path   = DB_FILES[choice]  # from your previous mapping

# now show what we’re using:
st.write(f"Using engine URL: `{engine_url}`")
if engine_url.startswith("sqlite"):
    st.write("File exists?", os.path.exists(raw_path))

# build engine once
sql_engine = sql.create_engine(
    engine_url,
    connect_args={"check_same_thread": False} if engine_url.startswith("sqlite") else {},
    poolclass=NullPool if engine_url.startswith("sqlite") else None,
)

