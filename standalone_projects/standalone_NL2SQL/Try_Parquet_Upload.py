from standalone_projects.standalone_NL2SQL.load_big_file import load_big_file_to_parquet, query_parquet
import streamlit as st

csv_path = "./data/all_reviews.csv"
parquet_path = "all_reviews.parquet"

load_big_file_to_parquet(csv_path, parquet_path)
df = query_parquet(parquet_path, limit=10000)

st.title('Big File Visualizer')
st.dataframe(df)