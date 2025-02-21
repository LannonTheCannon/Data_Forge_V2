import os
import streamlit as st
from pandasai import SmartDataframe
from pandasai.callbacks import BaseCallback
from pandasai.llm import OpenAI
from pandasai.responses.response_parser import ResponseParser
from data import load_data
from streamlit_ace import st_ace  # Streamlit code editor

import pandas as pd
import matplotlib.pyplot as plt

# Enable Wide Mode
st.set_page_config(layout="wide")


class StreamlitCallback(BaseCallback):
    def __init__(self, left_container) -> None:
        """Initialize callback handler."""
        self.left_container = left_container
        self.generated_code = ""

    def on_code(self, response: str):
        """Displays AI-generated code"""
        self.generated_code = response
        self.left_container.code(response, language="python")

    def get_generated_code(self):
        """Returns the last generated code"""
        return self.generated_code


class StreamlitResponse(ResponseParser):
    def __init__(self, context) -> None:
        super().__init__(context)

    def format_dataframe(self, result):
        st.dataframe(result["value"])

    def format_plot(self, result):
        """Temp Chart now appears in the left column below the code"""
        left_chart_container.image(result["value"])

    def format_other(self, result):
        st.write(result["value"])


st.write("# Chat with Credit Card Fraud Dataset 🦙")

# Load the dataset
df = load_data()

with st.expander("🔎 Dataframe Preview"):
    st.write(df.tail(3))

query = st.text_area("🗣️ Ask a Data Analysis Question:")

# Two Columns: Left (Generated Code & Chart), Right (Editable Code Editor & Chart Output)
left_column, right_column = st.columns([1, 1])

left_code_container = left_column.empty()  # AI-Generated Code
left_chart_container = left_column.empty()  # Temp Chart (below AI Code)

# Initialize session state for generated code
if "generated_code" not in st.session_state:
    st.session_state.generated_code = ""

# Right Column Layout
with right_column:
    # Code Editor Container (only initialized once)
    edited_code = st_ace(
        value=st.session_state.generated_code,
        language="python",
        theme="monokai",
        key="code_editor",  # Ensure a unique key
        height=800,
    )

    # "Run Code" Button
    run_button = st.button("Run Code")

    # Placeholder for Chart Output (Ensures it is BELOW the button)
    right_chart_container = st.empty()

if query:
    llm = OpenAI(api_token=os.environ["OPENAI_API_KEY"])
    callback_handler = StreamlitCallback(left_code_container)

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

    # Update session state with AI-generated code
    st.session_state.generated_code = generated_code

    # Update the Left Side Code Display
    left_code_container.code(generated_code, language="python")

if run_button:
    try:
        exec_locals = {}  # Isolated local execution

        # Execute User-Edited Code
        exec(edited_code, {"df": df, "pd": pd, "plt": plt, "st": st}, exec_locals)

        # Run function if it exists
        if "analyze_data" in exec_locals:
            result = exec_locals["analyze_data"]([df])  # Call function with DataFrame list

            # Place the chart **DIRECTLY BELOW the code editor**
            if result["type"] == "plot":
                right_chart_container.image(result["value"])
            elif result["type"] == "dataframe":
                right_chart_container.dataframe(result["value"])
            elif result["type"] == "string":
                right_chart_container.write(result["value"])
            elif result["type"] == "number":
                right_chart_container.write(f"Result: {result['value']}")

    except Exception as e:
        st.error(f"⚠️ Error running the code: {e}")
