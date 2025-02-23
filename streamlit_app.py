import os
import base64
import shutil
import openai
import streamlit as st
from streamlit_ace import st_ace
from pandasai import SmartDataframe
from pandasai.callbacks import BaseCallback
from pandasai.llm import OpenAI as PandasOpenAI
from pandasai.responses.response_parser import ResponseParser

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import inspect

from data import load_data

# ---------------------------------------------
# 1️⃣ Streamlit and OpenAI Setup
# ---------------------------------------------
st.set_page_config(layout="wide")

client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ---------------------------------------------
# 2️⃣ Session State Initialization
# ---------------------------------------------
if "generated_code" not in st.session_state:
    st.session_state.generated_code = ""

if "editor_code" not in st.session_state:
    st.session_state.editor_code = ""

if "chart_generated" not in st.session_state:
    st.session_state.chart_generated = False

if "chart_path" not in st.session_state:
    st.session_state.chart_path = ""

if "pandasai_query" not in st.session_state:
    st.session_state.pandasai_query = ""

# ✅ Store the GPT-4 Vision response (chart interpretation)
if "gpt4_vision_text" not in st.session_state:
    st.session_state.gpt4_vision_text = None

# ✅ Store the user-edited code's output
if "user_code_result" not in st.session_state:
    st.session_state.user_code_result = None


# ---------------------------------------------
# 3️⃣ Utility Functions
# ---------------------------------------------
# def load_data():
#     """Load your dataset here (example data)."""
#     data = {
#         "A": [1, 2, 3, 4, 5],
#         "B": [10, 20, 30, 40, 50],
#         "Fraud": [0, 1, 0, 1, 0]
#     }
#     return pd.DataFrame(data)


def encode_image(image_path):
    """Encodes an image to Base64 for OpenAI Vision API."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def analyze_chart_with_openai(image_path):
    """Sends the generated chart to OpenAI GPT-4 Vision API and returns insights."""
    try:
        if not os.path.exists(image_path):
            return "⚠️ No chart found. Please generate a chart first."

        base64_image = encode_image(image_path)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "This is a data visualization chart generated from a dataset related to credit card fraud detection. "
                                "Analyze the chart and provide insights by answering the following:\n\n"
                                "**1️⃣ Describe the chart:** What type of chart is this? What variables are on each axis?\n"
                                "**2️⃣ Identify trends & patterns:** Any notable trends, peaks, declines, clusters, or outliers?\n"
                                "**3️⃣ Interpret key findings:** What does it suggest about fraud detection?\n"
                                "**4️⃣ Provide actionable insights:** What recommendations or next steps would you suggest?\n\n"
                                "Be as detailed as possible and avoid assumptions beyond what the chart visually represents."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                        },
                    ],
                }
            ],
            max_tokens=500,
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"⚠️ Error analyzing chart: {e}"


# ---------------------------------------------
# 4️⃣ Callback Classes
# ---------------------------------------------
class StreamlitCallback(BaseCallback):
    """Displays AI-generated code."""

    def __init__(self, code_container, response_container) -> None:
        self.code_container = code_container
        self.response_container = response_container
        self.generated_code = ""

    def on_code(self, response: str):
        """Displays AI-generated code"""
        self.generated_code = response
        self.code_container.code(response, language="python")

    def get_generated_code(self):
        return self.generated_code


class StreamlitResponse(ResponseParser):
    """Handles DataFrames, Plots, and Other outputs.
       ❗️We do NOT display the chart inside this class to avoid double-loading.
    """

    def __init__(self, context) -> None:
        super().__init__(context)

    def format_dataframe(self, result):
        st.dataframe(result["value"])
        st.session_state.chart_generated = False

    def format_plot(self, result):
        """Save the chart to temp_chart.png, but don't display to avoid duplication."""
        chart_path = os.path.abspath("temp_chart.png")

        if isinstance(result["value"], str):
            existing_chart_path = os.path.abspath(result["value"])
            if existing_chart_path == chart_path:
                st.session_state.chart_generated = True
                st.session_state.chart_path = chart_path
            else:
                try:
                    shutil.copy(existing_chart_path, chart_path)
                    st.session_state.chart_generated = True
                    st.session_state.chart_path = chart_path
                except Exception as e:
                    st.error(f"⚠️ Error copying chart: {e}")
                    st.session_state.chart_generated = False

        elif isinstance(result["value"], plt.Figure):
            result["value"].savefig(chart_path)
            st.session_state.chart_generated = True
            st.session_state.chart_path = chart_path
        else:
            st.error("⚠️ Unexpected chart format returned.")
            st.session_state.chart_generated = False

    def format_other(self, result):
        st.markdown(f"### 📌 AI Insight\n\n{result['value']}")
        st.session_state.chart_generated = False


# ---------------------------------------------
# 5️⃣ Main App - Multi "Page" Navigation
# ---------------------------------------------
st.title("Chat with Credit Card Fraud Dataset 🦙")
page = st.sidebar.radio("Select a Page", ["PandasAI Insights", "Code Editor"])

df = load_data()

# ==========================
# PAGE 1: PandasAI Insights
# ==========================
if page == "PandasAI Insights":
    st.subheader("PandasAI Analysis & GPT-4 Vision")

    with st.expander("🔎 Dataframe Preview"):
        st.dataframe(df.head(5))

    # Let user enter a query
    st.session_state.pandasai_query = st.text_area(
        "🗣️ Ask a Data Analysis Question:",
        value=st.session_state.pandasai_query
    )

    # If user enters a query, run PandasAI
    if st.session_state.pandasai_query:
        # Create containers for code & response
        code_container = st.container()
        response_container = st.container()

        llm = PandasOpenAI(api_token=os.environ["OPENAI_API_KEY"])
        callback_handler = StreamlitCallback(code_container, response_container)

        query_engine = SmartDataframe(
            df,
            config={
                "llm": llm,
                "response_parser": StreamlitResponse,  # Class, not instance
                "callback": callback_handler,
            },
        )

        answer = query_engine.chat(st.session_state.pandasai_query)

        # Retrieve AI-generated code
        generated_code = callback_handler.get_generated_code()
        st.session_state.generated_code = generated_code
        st.session_state.editor_code = generated_code

    # ✅ If a chart has been generated, display it once.
    if st.session_state.chart_generated:
        st.image(st.session_state.chart_path, caption="Chart from PandasAI")

        # If we already have GPT-4 text, show it
        if st.session_state.gpt4_vision_text:
            st.markdown(f"### 📊 AI Chart Breakdown (Saved)\n\n{st.session_state.gpt4_vision_text}")
            # Button is now disabled since user already has the analysis
            st.button("🧐 GPT-4 Vision Interpreter", disabled=True)
        else:
            # Show active GPT-4 Vision button
            gpt4_vision_button = st.button("🧐 GPT-4 Vision Interpreter")
            if gpt4_vision_button:
                st.session_state.gpt4_vision_text = analyze_chart_with_openai(st.session_state.chart_path)
                st.markdown(f"### 📊 AI Chart Breakdown\n\n{st.session_state.gpt4_vision_text}")
    else:
        st.button("🧐 GPT-4 Vision Interpreter", disabled=True)

    if not st.session_state.generated_code and not st.session_state.chart_generated:
        st.info("Enter a query above to generate a chart with PandasAI.")

# ======================
# PAGE 2: Code Editor
# ======================
else:
    st.subheader("User Code Editor & Execution")

    # Show the user code if we have it
    if st.session_state.generated_code:
        edited_code = st_ace(
            value=st.session_state.editor_code,
            language="python",
            theme="tomorrow",
            key="code_editor",
            height=750,
        )
        st.session_state.editor_code = edited_code

        col1, col2 = st.columns([3, 1])
        with col1:
            st.write("")
        with col2:
            run_button = st.button("🚀 Run Code")

        if run_button:
            try:
                exec_locals = {}
                exec(
                    st.session_state.editor_code,
                    {"df": df, "pd": pd, "plt": plt, "st": st, "sns": sns},
                    exec_locals
                )

                if "analyze_data" in exec_locals:
                    analyze_func = exec_locals["analyze_data"]
                    params = inspect.signature(analyze_func).parameters

                    if "user_input" in params:
                        result = analyze_func([df], user_input="")
                    else:
                        result = analyze_func([df])

                    st.session_state.user_code_result = result

            except Exception as e:
                st.error(f"⚠️ Error running the code: {e}")

        # ✅ Show the user-code output if it exists
        if st.session_state.user_code_result:
            result = st.session_state.user_code_result
            if result["type"] == "plot":
                st.image(result["value"])
            elif result["type"] == "dataframe":
                st.dataframe(result["value"])
            elif result["type"] == "string":
                st.markdown(f"### 📌 AI Insight\n\n{result['value']}")
            elif result["type"] == "number":
                st.write(f"Result: {result['value']}")
    else:
        st.info("No generated code yet. Please go to 'PandasAI Insights' and enter a query first.")
