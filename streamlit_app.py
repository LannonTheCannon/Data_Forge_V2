import os
import streamlit as st
from pandasai import SmartDataframe
from pandasai.callbacks import BaseCallback
from pandasai.llm import OpenAI
from pandasai.responses.response_parser import ResponseParser
from data import load_data
from streamlit_ace import st_ace  # Streamlit code editor
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import inspect  # ✅ NEW: Check function signature dynamically

# Enable Wide Mode
st.set_page_config(layout="wide")


class StreamlitCallback(BaseCallback):
    def __init__(self, left_code_container, left_response_container) -> None:
        """Initialize callback handler."""
        self.left_code_container = left_code_container
        self.left_response_container = left_response_container
        self.generated_code = ""

    def on_code(self, response: str):
        """Displays AI-generated code"""
        self.generated_code = response
        self.left_code_container.code(response, language="python")

    def get_generated_code(self):
        """Returns the last generated code"""
        return self.generated_code


class StreamlitResponse(ResponseParser):
    def __init__(self, context) -> None:
        super().__init__(context)

    def format_dataframe(self, result):
        """Ensures DataFrames appear below the AI-generated code"""
        left_response_container.dataframe(result["value"])

    def format_plot(self, result):
        """Ensures Charts appear below the AI-generated code"""
        left_response_container.image(result["value"])

    def format_other(self, result):
        """Ensures Text responses appear properly formatted on the left side"""
        left_response_container.markdown(f"### 📌 AI Insight\n\n{result['value']}")


st.write("# Chat with Credit Card Fraud Dataset 🦙")

# Load the dataset
df = load_data()

with st.expander("🔎 Dataframe Preview"):
    st.write(df.tail(3))

query = st.text_area("🗣️ Ask a Data Analysis Question:")

# Two Columns: Left (Generated Code & Response), Right (Editable Code Editor & Chart Output)
left_column, right_column = st.columns([1, 1])

left_code_container = left_column.empty()  # AI-Generated Code
left_response_container = left_column.empty()  # ✅ Response Container (Chart/Text/Table)

# Initialize session state for generated code and editor text
if "generated_code" not in st.session_state:
    st.session_state.generated_code = ""

if "editor_code" not in st.session_state:
    st.session_state.editor_code = ""  # Ensure it's initialized

# 🚀 **Process User Query**
if query:
    llm = OpenAI(api_token=os.environ["OPENAI_API_KEY"])
    callback_handler = StreamlitCallback(left_code_container, left_response_container)

    query_engine = SmartDataframe(
        df,
        config={
            "llm": llm,
            "response_parser": StreamlitResponse,
            "callback": callback_handler,
        },
    )

    answer = query_engine.chat(query)
    generated_code = callback_handler.get_generated_code()

    # **Update session state with AI-generated code**
    st.session_state.generated_code = generated_code
    st.session_state.editor_code = generated_code  # **Sync with editor!**

# ✅ **Show the code editor & run button ONLY IF a query was submitted**
if st.session_state.generated_code:
    # ✅ Right Column Layout (Editor + Button Closer)
    with right_column:
        # Code Editor
        edited_code = st_ace(
            value=st.session_state.editor_code,
            language="python",
            theme="tomorrow",
            key="code_editor",  # Ensure key is static to persist state
            height=750,
        )

        # 🔹 Store edited code back to session state before running
        st.session_state.editor_code = edited_code

        # **Move Button Directly Under Editor (No Extra Gaps)**
        col1, col2 = st.columns([3, 1])  # Create button row (aligned right)
        with col1:
            st.write("")  # Placeholder for spacing
        with col2:
            run_button = st.button("🚀 Run Code")  # Aligns right & closer to editor

        # Placeholder for Chart Output **DIRECTLY BELOW**
        right_chart_container = st.empty()

    if run_button:
        try:
            exec_locals = {}  # Isolated local execution

            # 🟢 **Use the latest user-edited code**
            exec(st.session_state.editor_code, {
                "df": df, "pd": pd, "plt": plt, "st": st, "sns": sns
            }, exec_locals)

            # ✅ **Check function parameters before calling**
            if "analyze_data" in exec_locals:
                analyze_func = exec_locals["analyze_data"]
                params = inspect.signature(analyze_func).parameters  # Get function parameters

                if "user_input" in params:
                    result = analyze_func([df], user_input="")  # ✅ Pass default empty string
                else:
                    result = analyze_func([df])  # ✅ Call normally if 'user_input' isn’t required

                # Place the chart **DIRECTLY BELOW the code editor**
                if result["type"] == "plot":
                    right_chart_container.image(result["value"])
                elif result["type"] == "dataframe":
                    right_chart_container.dataframe(result["value"])
                elif result["type"] == "string":
                    right_chart_container.markdown(f"### 📌 AI Insight\n\n{result['value']}")
                elif result["type"] == "number":
                    right_chart_container.write(f"Result: {result['value']}")

        except Exception as e:
            st.error(f"⚠️ Error running the code: {e}")
