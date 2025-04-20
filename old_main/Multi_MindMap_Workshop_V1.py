import numpy as np
import openai
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import json
from code_editor import code_editor
import plotly.graph_objects as go
import plotly.io as pio
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_openai import ChatOpenAI
from ai_data_science_team.ds_agents import EDAToolsAgent
from ai_data_science_team import PandasDataAnalyst, DataWranglingAgent, DataVisualizationAgent
import streamlit as st
from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
from streamlit_flow.state import StreamlitFlowState
from streamlit_flow.layouts import ManualLayout, RadialLayout, TreeLayout
import random
# from standalone_projects.standalone_streamlit_flow_nodes.basic_streamlit_flow_nodes_4 import COLOR_PALETTE

st.set_page_config(page_title="Advanced PandasAI + Vision Demo", layout="wide")
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

COLOR_PALETTE = ["#FF6B6B", "#6BCB77", "#4D96FF", "#FFD93D", "#845EC2", "#F9A826"]

# ##################### SESSION STATE STUFF ####################### #

if "curr_state" not in st.session_state:
    # Prepare root node. We'll store the dataset metadata in "full_question" if we have it.
    dataset_label = st.session_state.get("dataset_name", "Dataset")

    root_theme = ThemeNode(
        node_id="S0",
        label="ROOT",
        full_question="Overview of the dataset",
        category="Meta",
        node_type="theme",
        parent_id=None,
        metadata={"content": dataset_label}
    )

    st.session_state.mindmap_nodes = {"S0": root_theme}
    st.session_state.curr_state = StreamlitFlowState(
        nodes=[root_theme.to_streamlit_node()],
        edges=[]
    )

for key in ["chart_path", "df", "df_preview", "df_summary", "metadata_string", "saved_charts", "DATA_RAW", "plots",
            "dataframes", "msg_index", "clicked_questions", "dataset_name"]:
    if key not in st.session_state:
        st.session_state[key] = None if key in ["chart_path", "df", "df_preview", "df_summary", "metadata_string", "DATA_RAW"] else []

if "expanded_nodes" not in st.session_state:
    st.session_state.expanded_nodes = set()

if "seen_embeddings" not in st.session_state:
    st.session_state.seen_embeddings = []