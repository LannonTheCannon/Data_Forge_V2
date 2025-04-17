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

# Setting up session state vars
for key in ["chart_path", "df", "df_preview", "df_summary", "metadata_string", "saved_charts", "DATA_RAW", "plots",
            "dataframes", "msg_index"]:
    if key not in st.session_state:
        st.session_state[key] = None if key in ["chart_path", "df", "df_preview", "df_summary", "metadata_string",
                                                "DATA_RAW"] else []

# Data Upload function
def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        return df
    return None

def get_assistant_interpretation(user_input, metadata, valid_columns):
    column_names = ', '.join(valid_columns)

    prompt = f"""
You are a visualization interpreter.

Your job is to rephrase the user's request into a **precise and code-compatible** instruction. Use this format:

→ "Create a [chart type] of the `[y_column]` on the y-axis ([aggregation]) and the `[x_column]` in the x-axis and make the chart [color]."

---

Rules:
- DO NOT invent or guess column names. Use ONLY from this list:
  {column_names}
- NEVER say "average salary in USD" — instead say: "`salary_in_usd` on the y-axis (avg)"
- Keep aggregation words like "avg", "sum", or "count" OUTSIDE of the column name.
- Keep axis mappings clear and exact.
- Mention the color explicitly at the end.
- Avoid words like “visualize” or “illustrate.” Just say "Create a bar chart..."

---

📥 USER QUERY:
{user_input}

📊 METADATA:
{metadata}

✍️ Respond with just one sentence using the format shown above.
"""

    try:
        response = openai.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that rewrites data visualization queries into precise and code-friendly instructions."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.2,
        )
        return response.choices[0].message.content

    except Exception as e:
        st.warning(f"Error in get_assistant_interpretation: {e}")
        return "Could not interpret user request."
# def get_assistant_interpretation(user_input, metadata, valid_columns):
#     column_names = ', '.join(valid_columns)
#
#     prompt = f"""
#     *Reinterpret the user’s request into a clear, visualization-ready question that aligns with the dataset’s
#     structure and is optimized for charting.
#
#     User query: {user_input}
#
#     These are the ONLY valid column names in the dataset. If the user asks for "average salary", you must calculate it using `.groupby()` on `job_title` and take the mean of `salary_in_usd`. Do NOT reference non-existent columns.
#
#     Dataset metadata:
#     {metadata}
#
#     ** IMPORTANT **
#     1) Extract the exact column names from the user's query
#     2) understand the key function of the user's intent and side functions of the user's query
#
#     For example:
#     "create a plot of the average salary in usd by job title in a bar chart"
#
#     You will find that there is no average_salary_in_usd but there is "salary_in_usd" and you will
#     need to get the mean of that. You'll also see that job_title is a column that is found in the
#     dataset.
#
#     You must ONLY use column names listed above. DO NOT guess or create new ones.
#
#     Return a reframed question that will guide a charting AI to produce the correct visualization.
#
#     Your reframed question should look like this
#
#     "create a bar chart of the [salary_in_usd] on the y-axis but get the avg and the [job_title] on the x-axis.
#     """
#
#     try:
#         # Again, use the Chat endpoint for a chat model (like gpt-3.5-turbo)
#         response = openai.chat.completions.create(
#             model="gpt-4.1-mini",
#             messages=[
#                 {"role": "system",
#                  "content": "You are a helpful data analysis assistant designed to extract the core intent of the user query and form a high value prompt that can be used 100% of the time for ai_data_science_team (visual agent) for charting code. "},
#                 {"role": "user", "content": prompt}
#             ],
#             max_tokens=300,
#             temperature=0.0,
#         )
#         summary = response.choices[0].message.content
#
#         return summary
#
#     except Exception as e:
#         st.warning(f"Error in get_assistant_interpretation: {e}")
#         return "Could not interpret user request."

def display_chat_history():
    if "chat_artifacts" not in st.session_state:
        st.session_state["chat_artifacts"] = {}

    for i, msg in enumerate(msgs.messages):
        role_label = "User" if msg.type == "human" else "Assistant"
        with st.chat_message(msg.type):
            st.markdown(f"**{role_label}:** {msg.content}")

            if i in st.session_state["chat_artifacts"]:
                for j, artifact in enumerate(st.session_state["chat_artifacts"][i]):
                    unique_key = f"msg_{i}_artifact_{j}"
                    editor_key = f"editor_code_{unique_key}"
                    output_key = f"output_chart_{unique_key}"

                    with st.expander(f"\U0001F4CE {artifact['title']}", expanded=True):
                        tabs = st.tabs(["📊 Output", "💻 Code"])

                        # --- Code Tab First, to capture edits and trigger updates ---
                        with tabs[1]:
                            code_before = st.session_state.get(editor_key, artifact.get("code", ""))
                            editor_response = code_editor(
                                code=code_before,
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

                            new_code = editor_response.get("text", "").strip()

                            # Only run if the code has changed
                            if new_code and new_code != st.session_state.get(editor_key):
                                try:
                                    exec_globals = {
                                        "df": st.session_state.df,
                                        "pd": pd,
                                        "np": np,
                                        "sns": sns,
                                        "go": go,
                                        "plt": plt,
                                        "pio": pio,
                                        "st": st,
                                        "json": json
                                    }
                                    exec_locals = {}
                                    exec(new_code, exec_globals, exec_locals)

                                    output_obj = exec_locals.get("fig") or \
                                                 exec_locals.get("output") or \
                                                 exec_locals.get("fig_dict")

                                    if isinstance(output_obj, dict) and "data" in output_obj and "layout" in output_obj:
                                        output_obj = pio.from_json(json.dumps(output_obj))

                                    artifact["data"] = output_obj
                                    artifact["render_type"] = "plotly" if isinstance(output_obj, go.Figure) else "dataframe"
                                    st.session_state[editor_key] = new_code
                                    st.session_state[output_key] = output_obj

                                except Exception as e:
                                    st.error(f"Error executing code: {e}")

                        # --- Output Tab ---
                        with tabs[0]:
                            output_obj = st.session_state.get(output_key, artifact.get("data"))
                            render_type = artifact.get("render_type")

                            if isinstance(output_obj, dict) and "data" in output_obj and "layout" in output_obj:
                                output_obj = pio.from_json(json.dumps(output_obj))

                            if render_type == "plotly":
                                st.plotly_chart(
                                    output_obj,
                                    use_container_width=True,
                                    config={
                                        "displayModeBar": True,
                                        "scrollZoom": True,
                                        "displaylogo": False
                                    },
                                    key=f"plotly_{output_key}"
                                )
                            elif render_type == "dataframe":
                                st.dataframe(output_obj, key=f"df_{output_key}")
                            else:
                                st.write(output_obj)

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
            msgs.add_ai_message("IMPORTANT: For best results use this formula -> Create a [chart] of the [field] on the y-axis (aggregation) and the [field] on the x-axis and make the chart [color].")
        if 'pandas_data_analyst' not in st.session_state:
            model = ChatOpenAI(model='gpt-4.1', api_key=st.secrets['OPENAI_API_KEY'])
            st.session_state.pandas_data_analyst = PandasDataAnalyst(
                model=model,
                data_wrangling_agent=DataWranglingAgent(model=model,
                                                        log=True,
                                                        n_samples=100),
                data_visualization_agent=DataVisualizationAgent(
                                                        model=model,
                                                        log=True,
                                                        log_path="logs",
                                                        overwrite=False,  # ✅ Ensures every chart gets a separate file
                                                        n_samples=100,
                                                        bypass_recommended_steps=False)
            )
        question = st.chat_input('Ask a question about your dataset!')
        interpretation = get_assistant_interpretation(
            question,
            st.session_state['metadata_string'],
            st.session_state.df.columns  # ✅ pass real column names
        )
        # print(interpretation)

        if question:
            msgs.add_user_message(interpretation)
            with st.spinner("Thinking..."):
                try:
                    st.session_state.pandas_data_analyst.invoke_agent(
                        user_instructions=interpretation,
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
                        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                        viz_code = result.get('data_visualization_function', "")
                        wrangle_code = result.get('data_wrangler_function', "")

                        # Combine both functions into one code block
                        combined_code = f"{wrangle_code}\n\n{viz_code}\n\n# Runtime Execution\noutput = data_visualization(data_wrangler([df]))"

                        st.session_state["chat_artifacts"][msg_index].append({
                            "title": "Chart",
                            "render_type": "plotly",
                            "data": plot_obj,
                            "code": combined_code
                        })
                        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                        print(result['data_visualization_function'])

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
