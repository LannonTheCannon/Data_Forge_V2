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
import json
from code_editor import code_editor
import traceback
from pandasai import SmartDataframe
from pandasai.llm import OpenAI as PandasOpenAI
from pandasai.callbacks import BaseCallback
from pandasai.responses.response_parser import ResponseParser
import plotly.graph_objects as go
import plotly.io as pio
from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
from streamlit_flow.state import StreamlitFlowState
from streamlit_flow.layouts import ManualLayout, RadialLayout, TreeLayout
import random
from uuid import uuid4
import streamlit.components.v1 as components
from pathlib import Path
import html
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_openai import ChatOpenAI
from ai_data_science_team.ds_agents import EDAToolsAgent
from ai_data_science_team.utils.matplotlib import matplotlib_from_base64
from ai_data_science_team.utils.plotly import plotly_from_dict
from ai_data_science_team import PandasDataAnalyst, DataWranglingAgent, DataVisualizationAgent

st.set_page_config(page_title="Advanced PandasAI + Vision Demo", layout="wide")

client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Seeting up session state vars
for key in ["chart_path", "df", "df_preview", "df_summary", "metadata_string", "saved_charts", "DATA_RAW", "plots", "dataframes", "msg_index"]:
    if key not in st.session_state:
        st.session_state[key] = None if key in ["chart_path", "df", "df_preview", "df_summary", "metadata_string", "DATA_RAW"] else []

# Data Upload function
def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        return df
    return None

def plotly_from_dict(plotly_dict):

    if isinstance(plotly_dict, dict) and "data" in plotly_dict and "layout" in plotly_dict:

        return pio.from_json(json.dumps(plotly_dict))

    return None


def display_chat_history():
    if "chat_artifacts" not in st.session_state:
        st.session_state["chat_artifacts"] = {}

    for i, msg in enumerate(msgs.messages):
        role_label = "User" if msg.type == "human" else "Assistant"
        with st.chat_message(msg.type):
            st.markdown(f"**{role_label}:** {msg.content}")

            if i in st.session_state["chat_artifacts"]:
                for j, artifact in enumerate(st.session_state["chat_artifacts"][i]):
                    with st.expander(f"\U0001F4CE {artifact['title']}", expanded=True):
                        tabs = st.tabs(["\U0001F4CA Output", "💻 Code"])

                        with tabs[0]:
                            render_type = artifact.get("render_type")
                            data = artifact.get("data")
                            if isinstance(data, dict) and "data" in data and "layout" in data:
                                data = pio.from_json(json.dumps(data))
                            fig = plotly_from_dict(data)

                            if fig is not None:

                                st.plotly_chart(fig, use_container_width=True, config={

                                    "displayModeBar": True,

                                    "scrollZoom": True,

                                    "displaylogo": False

                                })
                            elif render_type == "dataframe":
                                st.dataframe(data)
                            elif render_type == "string":
                                st.markdown(data)
                            elif render_type == "number":
                                st.write(data)
                            else:
                                st.warning("Unknown artifact type.")

                        unique_key = f"msg_{i}_artifact_{j}"

                        with tabs[1]:
                            editor_response = code_editor(
                                code=artifact.get("code", "# No code available"),
                                lang="python",
                                theme="dracula",
                                height=300,
                                buttons=[
                                    {
                                        "name": "Run",
                                        "feather": "Play",
                                        "primary": True,
                                        "hasText": True,
                                        "showWithIcon": True,
                                        "commands": ["submit"],
                                        "style": {"bottom": "0.44rem", "right": "0.4rem"}
                                    }
                                ],
                                key=f"code_editor_{unique_key}"
                            )

                            code_to_run = editor_response.get("text", "").strip()

                            if code_to_run:

                                try:

                                    exec_globals = {

                                        "df": st.session_state.df,

                                        "pd": pd,

                                        "np": np,

                                        "go": go,

                                        "plt": plt,

                                        "pio": pio,

                                        "json": json,

                                        "st": st

                                    }

                                    exec_locals = {}

                                    # Execute the code

                                    exec(code_to_run, exec_globals, exec_locals)

                                    # Get the figure object

                                    fig = exec_locals.get('fig') or exec_locals.get('output')

                                    if fig is not None:

                                        if isinstance(fig, go.Figure):

                                            # Convert to JSON and back to ensure consistent format

                                            fig_json = pio.to_json(fig)

                                            fig_dict = json.loads(fig_json)

                                            # Update the artifact data

                                            st.session_state["chat_artifacts"][i][j]["data"] = fig_dict

                                            st.session_state["chat_artifacts"][i][j]["render_type"] = "plotly"

                                            st.session_state["chat_artifacts"][i][j]["code"] = code_to_run

                                            st.experimental_rerun()



                                        elif isinstance(fig, pd.DataFrame):

                                            st.session_state["chat_artifacts"][i][j]["data"] = fig

                                            st.session_state["chat_artifacts"][i][j]["render_type"] = "dataframe"

                                            st.session_state["chat_artifacts"][i][j]["code"] = code_to_run

                                            st.experimental_rerun()

                                    else:

                                        st.warning(
                                            "No output detected. Make sure to assign your result to 'fig' or 'output' variable.")



                                except Exception as e:

                                    st.error(f"Error executing code: {str(e)}\n{traceback.format_exc()}")

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
            df = load_data(uploaded_file)
            if df is not None:
                st.session_state.df = df
                st.session_state["DATA_RAW"] = df
                st.session_state.df_preview = df.head()
                st.session_state.df_summary = df.describe()
                dataset_name = uploaded_file.name.rsplit('.', 1)[0]
                st.session_state['dataset_name'] = dataset_name
                st.write(dataset_name)

                if st.session_state.df_summary is not None:
                    cols = list(st.session_state.df_summary.columns)
                    row_count = st.session_state.df.shape[0]
                    st.session_state.metadata_string = (
                        f"Columns: {cols}\n"
                        f"Total Rows: {row_count}\n"
                        f"Summary Stats:\n{st.session_state.df_summary}"
                    )

        if st.session_state.df is not None:
            st.write("### Data Preview")
            st.write(st.session_state.df_preview)
            st.write("### Data Summary")
            st.write(st.session_state.df_summary)

    elif page == 'Data Analyst':
        st.subheader('Pandas Data Analyst Mode')

        msgs = StreamlitChatMessageHistory(key="pandas_data_analyst_messages")
        if len(msgs.messages) == 0:
            msgs.add_ai_message("Hey whatsup! I'm your personal data assistant. I can create tables or graphs for you.")

        if 'pandas_data_analyst' not in st.session_state:
            model = ChatOpenAI(model='gpt-4o-mini', api_key=st.secrets['OPENAI_API_KEY'])
            st.session_state.pandas_data_analyst = PandasDataAnalyst(
                model=model,
                data_wrangling_agent=DataWranglingAgent(model=model, log=False, n_samples=100),
                data_visualization_agent=DataVisualizationAgent(model=model, log=False, n_samples=100)
            )

        question = st.chat_input('Ask a question about your dataset!')
        if question:
            msgs.add_user_message(question)
            with st.spinner("Thinking..."):
                try:
                    st.session_state.pandas_data_analyst.invoke_agent(
                        user_instructions=question,
                        data_raw=st.session_state["DATA_RAW"]
                    )
                    result = st.session_state.pandas_data_analyst.get_response()
                    route = result.get("routing_preprocessor_decision", "")

                    ai_msg = "Here's what I found:"
                    msgs.add_ai_message(ai_msg)
                    msg_index = len(msgs.messages) - 1

                    if "chat_artifacts" not in st.session_state:
                        st.session_state["chat_artifacts"] = {}

                    st.session_state["chat_artifacts"][msg_index] = []

                    if route == "chart" and not result.get("plotly_error", False):
                        plot_obj = pio.from_json(json.dumps(result["plotly_graph"]))
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

        display_chat_history()

    else:
        st.info("Data Storytelling page coming soon!")
