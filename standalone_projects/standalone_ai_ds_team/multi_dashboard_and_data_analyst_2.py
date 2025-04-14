import json
import traceback
from types import ModuleType
from uuid import uuid4

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import streamlit as st
from code_editor import code_editor
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_openai import ChatOpenAI

# 👉 your own package imports ---------------------------------------------------
from ai_data_science_team import (
    PandasDataAnalyst,
    DataWranglingAgent,
    DataVisualizationAgent,
)

# =============================================================================
# 🌐 Page config
# =============================================================================
st.set_page_config(page_title="Advanced PandasAI + Vision Demo", layout="wide")

# =============================================================================
# 🗄️ Session state defaults
# =============================================================================
DEFAULTS = {
    "df": None,
    "df_preview": None,
    "df_summary": None,
    "DATA_RAW": None,
    "dataset_name": "",
    "artifacts": {},          # {artifact_id: {...}}
    "chat_artifact_ids": {},  # {msg_index: [artifact_id, ...]}
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

# =============================================================================
# 📥 File loader
# =============================================================================

def load_data(uploaded_file):
    if uploaded_file is None:
        return None
    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)

# =============================================================================
# 🧩 User‑code runner with auto‑capture of st.plotly_chart / st.dataframe
# =============================================================================

def _infer_render_type(obj):
    if isinstance(obj, go.Figure):
        return "plotly"
    if isinstance(obj, pd.DataFrame):
        return "dataframe"
    if isinstance(obj, (int, float)):
        return "number"
    return "string"


class _StProxy(ModuleType):
    """Proxy around the real streamlit module that records what the user shows."""

    def __init__(self, real):
        super().__init__(real.__name__)
        self._real = real
        self._captured = []  # list of (obj, render_type)

    # Capture Plotly
    def plotly_chart(self, fig, *args, **kwargs):
        self._captured.append((fig, "plotly"))
        return self._real.plotly_chart(fig, *args, **kwargs)

    # Capture DataFrame
    def dataframe(self, df, *args, **kwargs):
        self._captured.append((df, "dataframe"))
        return self._real.dataframe(df, *args, **kwargs)

    def __getattr__(self, item):
        return getattr(self._real, item)


def run_user_code(code_str: str, exec_globals: dict | None = None):
    """Run *code_str* and return (output, render_type).

    Accepted ways for the user to produce output:
      • assign to `output`  ➜  `output = fig`
      • assign to `fig` / `df`
      • call `st.plotly_chart(fig)` / `st.dataframe(df)` (auto‑captured)
    """
    if exec_globals is None:
        exec_globals = {}

    # Safe context
    safe_ctx = dict(pd=pd, np=np, go=go, plt=plt)
    exec_globals.update(safe_ctx)

    # Patch streamlit inside the user sandbox
    st_proxy = _StProxy(st)
    exec_globals["st"] = st_proxy

    exec_locals: dict = {}
    try:
        exec(code_str, exec_globals, exec_locals)
    except Exception:
        raise RuntimeError(traceback.format_exc())

    # 1️⃣ explicit variable
    for name in ("output", "fig", "df"):
        if name in exec_locals:
            obj = exec_locals[name]
            return obj, _infer_render_type(obj)

    # 2️⃣ captured via st proxy (return the last one)
    if st_proxy._captured:
        return st_proxy._captured[-1]

    # 3️⃣ fallback scan
    for obj in exec_locals.values():
        rt = _infer_render_type(obj)
        if rt != "string" or isinstance(obj, str):
            return obj, rt

    raise RuntimeError("Nothing was displayed. Create a Plotly figure or DataFrame, assign it to a variable, or call st.plotly_chart/ st.dataframe inside your code.")

# =============================================================================
# 🖼️ Artifact block (Output & Code tabs)
# =============================================================================

def artifact_block(artifact_id: str):
    art = st.session_state.artifacts[artifact_id]
    st.markdown(f"#### {art['title']}")

    with st.expander("", expanded=True):
        tab_out, tab_code = st.tabs(["📊 Output", "💻 Code"])

        with tab_out:
            rt, data = art["render_type"], art["data"]
            if rt == "plotly":
                st.plotly_chart(data, use_container_width=True)
            elif rt == "dataframe":
                st.dataframe(data)
            elif rt == "string":
                st.markdown(data)
            elif rt == "number":
                st.write(data)
            else:
                st.warning(f"Unknown render type: {rt}")

        with tab_code:
            editor = code_editor(
                code=art["code"],
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
                        "style": {"bottom": "0.44rem", "right": "0.4rem"},
                    }
                ],
                key=f"code_editor_{artifact_id}",
            )

            new_code = editor.get("text", art["code"]).strip()
            if new_code and new_code != art["code"]:
                try:
                    new_output, new_rt = run_user_code(new_code, exec_globals={"df": st.session_state.df})
                    st.session_state.artifacts[artifact_id] = {
                        **art,
                        "code": new_code,
                        "data": new_output,
                        "render_type": new_rt,
                    }
                    st.experimental_rerun()
                except Exception as e:
                    st.error(e)

# =============================================================================
# 💬 Chat history renderer
# =============================================================================

def display_chat_history(msgs: StreamlitChatMessageHistory):
    for i, msg in enumerate(msgs.messages):
        role = "User" if msg.type == "human" else "Assistant"
        with st.chat_message(msg.type):
            st.markdown(f"**{role}:** {msg.content}")
            for art_id in st.session_state.chat_artifact_ids.get(i, []):
                artifact_block(art_id)

# =============================================================================
# 📄 Routing
# =============================================================================
PAGE_OPTIONS = ["Data Upload", "Data Analyst", "Data Storytelling"]
page = st.sidebar.radio("Select a Page", PAGE_OPTIONS)

# =============================================================================
# 📈 Data Upload page
# =============================================================================
if page == "Data Upload":
    st.title("Upload your own Dataset!")
    uploaded_file = st.file_uploader("Upload CSV or Excel here", type=["csv", "xlsx", "xls"])

    if uploaded_file:
        df = load_data(uploaded_file)
        if df is not None:
            st.session_state.df = df
            st.session_state.DATA_RAW = df
            st.session_state.df_preview = df.head()
            st.session_state.df_summary = df.describe()
            st.session_state.dataset_name = uploaded_file.name.rsplit(".", 1)[0]

    if st.session_state.df is not None:
        st.write("### Data Preview")
        st.write(st.session_state.df_preview)
        st.write("### Data Summary")
        st.write(st.session_state.df_summary)

# =============================================================================
# 🤖 Data Analyst page
# =============================================================================
elif page == "Data Analyst":
    st.subheader("Pandas Data Analyst Mode")

    msgs = StreamlitChatMessageHistory(key="pandas_data_analyst_messages")
    if not msgs.messages:
        msgs.add_ai_message("Hey there! I'm your personal data assistant. I can create tables or graphs for you.")

    if "pandas_data_analyst" not in st.session_state:
        model = ChatOpenAI(model="gpt-4o-mini", api_key=st.secrets["OPENAI_API_KEY"])
        st.session_state.pandas_data_analyst = PandasDataAnalyst(
            model=model,
            data_wrangling_agent=DataWranglingAgent(model=model, log=False, n_samples=100),
            data_visualization_agent=DataVisualizationAgent(model=model, log=False, n_samples=100),
        )

    question = st.chat_input("Ask a question about your dataset!")
    if question:
        msgs.add_user_message(question)
        with st.spinner("Thinking..."):
            try:
                st.session_state.pandas_data_analyst.invoke_agent(
                    user_instructions=question, data_raw=st.session_state.DATA_RAW
                )
                result = st.session_state.pandas_data_analyst.get_response()
                route = result.get("routing_preprocessor_decision", "")

                msgs.add_ai_message("Here's what I found:")
                msg_index = len(msgs.messages) - 1
                st.session_state.chat_artifact_ids.setdefault(msg_index, [])

                if route == "chart" and not result.get("plotly_error", False):
                    from plotly.io import from_json

                    fig = from_json(json.dumps(result["plotly_graph"]))
                    artifact_id = str(uuid4())
                    st.session_state.artifacts[artifact_id] = {
                        "title": "Chart",
                        "render_type": "plotly",
                        "data": fig,
                        "code": result["data_visualization_function"],
                    }
                    st.session_state.chat_artifact_ids[msg_index].append(artifact_id)

                elif route == "table":
                    df_out = result.get("data_wrangled")
                    if df_out is not None:
                        artifact_id = str(uuid4())
                        st.session_state.artifacts[artifact_id] = {
                            "title": "Table",
                            "render_type": "dataframe",
                            "data": df_out,
                            "code": result["data_wrangler_function"],
                        }
                        st.session_state.chat_artifact_ids[msg_index].append(artifact_id)
            except Exception as e:
                msgs.add_ai_message(f"Error: {e}")

    display_chat_history(msgs)

# =============================================================================
# 📖 Placeholder page
# =============================================================================
else:
    st.info("Data Storytelling page coming soon!")
