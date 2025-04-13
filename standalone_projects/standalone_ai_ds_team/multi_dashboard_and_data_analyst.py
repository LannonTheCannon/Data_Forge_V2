import os
import time
import base64
import numpy as np
import shutil
import io
import openai
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_ace import st_ace
import inspect
from streamlit_elements import elements, dashboard, mui, html
# from data import load_data
import json
# ------------------------------
# PandasAI + Callbacks
# ------------------------------
from pandasai import SmartDataframe
from pandasai.llm import OpenAI as PandasOpenAI
from pandasai.callbacks import BaseCallback
from pandasai.responses.response_parser import ResponseParser
# ------------------------------
# Streamlit FLOW Layout
# ------------------------------
import streamlit as st
from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
from streamlit_flow.state import StreamlitFlowState
from streamlit_flow.layouts import ManualLayout, RadialLayout, TreeLayout
import random
from uuid import uuid4
# ------------------------------------------------------------------

# Importing data science team stuff
import streamlit.components.v1 as components
from pathlib import Path
import html
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_openai import ChatOpenAI
from ai_data_science_team.ds_agents import EDAToolsAgent
from ai_data_science_team.utils.matplotlib import matplotlib_from_base64
from ai_data_science_team.utils.plotly import plotly_from_dict

from ai_data_science_team import PandasDataAnalyst, DataWranglingAgent, DataVisualizationAgent

##############
# 1) Data Upload
# 2) Data Analyst
# 3) Dashboard

st.set_page_config(page_title="Advanced PandasAI + Vision Demo", layout="wide")

# ------------------------------
# OpenAI Setup
# ------------------------------
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

layout = [
    # (element_identifier, x, y, w, h, additional_props...)
    dashboard.Item("first_item", 0, 0, 2, 2),  # Draggable & resizable by default
    dashboard.Item("second_item", 2, 0, 2, 2),  # Not draggable
    dashboard.Item("third_item", 0, 2, 1, 1),    # Not resizable
    dashboard.Item("chart_item", 4, 0, 3, 3)   # Our new chart card
]

if "chart_path" not in st.session_state:
    st.session_state.chart_path = None
if 'df' not in st.session_state:
    st.session_state.df = None
if "df_preview" not in st.session_state:
    st.session_state.df_preview = None
if "df_summary" not in st.session_state:
    st.session_state.df_summary = None
if "metadata_string" not in st.session_state:
    st.session_state['metadata_string'] = ""
if "saved_charts" not in st.session_state:
    st.session_state['saved_charts'] = []
if "DATA_RAW" not in st.session_state:
    st.session_state["DATA_RAW"] = None
if 'plots' not in st.session_state:
    st.session_state.plots = []
if 'dataframes' not in st.session_state:
    st.session_state.dataframes = []

# ------------------ STEP 1 DATA UPLOAD -------------------#
def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        return df
    return None

# ------------------- STEP 2 Data Analyst Logic -------------------#
def display_chat_history():
    if "chat_artifacts" not in st.session_state:
        st.session_state["chat_artifacts"] = {}

    for i, msg in enumerate(msgs.messages):
        role_label = "User" if msg.type == "human" else "Assistant"
        with st.chat_message(msg.type):
            st.markdown(f"**{role_label}:** {msg.content}")
            if i in st.session_state["chat_artifacts"]:
                for artifact in st.session_state["chat_artifacts"][i]:
                    with st.expander(f"📎 {artifact['title']}", expanded=True):
                        tabs = st.tabs(["📊 Output", "💻 Code"])
                        with tabs[0]:
                            if artifact["render_type"] == "plotly":
                                st.plotly_chart(artifact["data"])
                            elif artifact["render_type"] == "dataframe":
                                st.dataframe(artifact["data"])
                            else:
                                st.write("Unknown artifact type.")
                        with tabs[1]:
                            st.code(artifact.get("code", "# No code available"), language="python")


# ------------------ Streamlit Multi Page Options -------------------#
PAGE_OPTIONS = [
    'Data Upload',
    'Data Analyst',
    'Data Storytelling'
]

page = st.sidebar.radio('Select a Page', PAGE_OPTIONS)

if __name__ == "__main__":
    if page == 'Data Upload':
        st.title('Upload your own Dataset!')
        uploaded_file = st.file_uploader('Upload CSV or Excel here', type=['csv', 'excel'])

        if uploaded_file is not None:
            # Load data into session state
            df = load_data(uploaded_file)
            if df is not None:
                st.session_state.df = df
                st.session_state["DATA_RAW"] = df
                st.session_state.df_preview = df.head()
                st.session_state.df_summary = df.describe()

                # Save dataset name without extension
                dataset_name = uploaded_file.name.rsplit('.', 1)[0]
                st.session_state['dataset_name'] = dataset_name
                st.write(dataset_name)

                # Rebuild a 'metadata_string' for the root node
                if st.session_state.df_summary is not None:
                    # Basic example of turning summary + columns into a string
                    cols = list(st.session_state.df_summary.columns)
                    row_count = st.session_state.df.shape[0]
                    st.session_state.metadata_string = (
                        f"Columns: {cols}\n"
                        f"Total Rows: {row_count}\n"
                        f"Summary Stats:\n{st.session_state.df_summary}"
                    )

        # Display preview & summary if data exists
        if st.session_state.df is not None:
            st.write("### Data Preview")
            st.write(st.session_state.df_preview)

            st.write("### Data Summary")
            st.write(st.session_state.df_summary)

    elif page == 'Data Analyst':
        st.subheader('Pandas Data Analyst Mode')

        # 1. Initialize chat message history
        msgs = StreamlitChatMessageHistory(key="pandas_data_analyst_messages")

        if len(msgs.messages) == 0:
            msgs.add_ai_message("Hey whatsup! I\'m your personally data assistant. I can create tables or graphs for you")

        # 2. Initialize the analyst agent if not already
        if 'pandas_data_analyst' not in st.session_state:
            model = ChatOpenAI(model='gpt-4o-mini', api_key=st.secrets['OPENAI_API_KEY'])
            st.session_state.pandas_data_analyst = PandasDataAnalyst(
                model=model,
                data_wrangling_agent=DataWranglingAgent(model=model,
                                                        log=False,
                                                        n_samples=100),
                data_visualization_agent=DataVisualizationAgent(model=model,
                                                                log=False,
                                                                n_samples=100))

        # Get the user input
        question = st.chat_input('Ask a question about your dataset!')
        if question:
            msgs.add_user_message(question)
            with st.spinner("Thinking..."):
                try:
                    # Run the agent
                    st.session_state.pandas_data_analyst.invoke_agent(
                        user_instructions=question,
                        data_raw=st.session_state["DATA_RAW"]
                    )
                    result = st.session_state.pandas_data_analyst.get_response()
                    route = result.get("routing_preprocessor_decision", "")

                    # Add AI message
                    ai_msg = "Here's what I found:"
                    msgs.add_ai_message(ai_msg)
                    msg_index = len(msgs.messages) - 1

                    # Store artifacts
                    if "chat_artifacts" not in st.session_state:
                        st.session_state["chat_artifacts"] = {}

                    st.session_state["chat_artifacts"][msg_index] = []
                    if route == "chart" and not result.get("plotly_error", False):
                        from plotly.io import from_json
                        plot_obj = from_json(json.dumps(result["plotly_graph"]))
                        st.session_state.plots.append(plot_obj)
                        st.session_state["chat_artifacts"][msg_index].append({
                            "title": "Chart",
                            "render_type": "plotly",
                            "data": plot_obj,
                            'code': result.get('data_visualization_function')
                        })

                    elif route == "table":
                        df = result.get("data_wrangled")
                        if df is not None:
                            st.session_state.dataframes.append(df)
                            st.session_state["chat_artifacts"][msg_index].append({
                                "title": "Table",
                                "render_type": "dataframe",
                                "data": df,
                                'code': result.get('data_wrangler_function')
                            })

                except Exception as e:
                    error_msg = f"Error: {e}"
                    msgs.add_ai_message(error_msg)

        # Display all messages and artifacts
        display_chat_history()