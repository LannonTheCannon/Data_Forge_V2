import sweetviz as sv
import pandas as pd
import streamlit as st
# import numpy as np
import warnings

# warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
# def load_data(uploaded_file):
#     """Load CSV/Excel into a DataFrame."""
#     if uploaded_file is not None:
#         df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
#         return df
#     return None

# Ensure Sweetviz is working correctly
def generate_sweetviz_report(data: pd.DataFrame):
    try:
        report = sv.analyze(data)
        report.show_html("sweetviz_report.html")  # Save the report as an HTML file
        return "sweetviz_report.html"  # Return the file path to the artifact
    except Exception as e:
        st.error(f"Error generating Sweetviz report: {e}")
        return None

uploaded_file = st.file_uploader("📂 Upload your dataset", type=["csv", "xlsx"])

if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    generate_sweetviz_report(df)


